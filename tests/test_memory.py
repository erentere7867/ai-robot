"""Tests for circulating hybrid memory: short-term scratchpad and long-term SQLite vector memory."""

from __future__ import annotations

import time

import numpy as np

from robot.agent.api import Pose
from robot.mind.memory import (
    HALF_LIFE,
    Memory,
    PersonProfile,
    TaskScratchpad,
    VisualRingBuffer,
    cosine_similarity,
    embed_text,
    freshness,
)
from robot.perception.tracker import Detection, Tracker


def _seen(tracker: Tracker, label: str, x: float, y: float, color: str = "", **kw) -> None:
    now = time.monotonic()
    for k in range(2):
        tracker.update([Detection(label, x, y, 0.1, color, score=0.8, **kw)], now + k * 0.1)


def test_what_was_seen_told_and_shown_comes_back_after_a_restart(tmp_path):
    before = Tracker()
    _seen(before, "keys", 3.8, -2.0, "silver")
    _seen(before, "person", 0.0, 1.0)  # people move: not remembered
    gone = before.objects()[0]
    _seen(before, "cup", 1.0, 1.0, "red")
    before.verify(next(t for t in before.objects() if t.label == "cup"), None)  # the vision model saw nothing there
    Memory(tmp_path / "home.db").save(
        before,
        ["the keys go in the bowl"],
        {"start": Pose(0, 0, 0), "kitchen door": Pose(2.0, 0.5, 90.0)},
    )

    after, memory = Tracker(), Memory(tmp_path / "home.db")  # a new process: nothing in it yet
    assert memory.restore(after) == 1
    (keys,) = after.objects()
    assert (keys.label, keys.color, keys.id) == ("keys", "silver", gone.id)
    assert (keys.x, keys.y) == (3.8, -2.0) and time.monotonic() - keys.last_seen < 5  # its age came back too
    assert memory.notes() == ["the keys go in the bowl"] and memory.places() == {"kitchen door": (2.0, 0.5, 90.0)}
    _seen(after, "ball", -1.0, 2.0)  # new things get new ids
    assert len({t.id for t in after.objects()}) == 2


def test_a_thing_found_gone_is_not_remembered_as_there(tmp_path):
    tracker, memory = Tracker(), Memory(tmp_path / "home.db")
    _seen(tracker, "keys", 3.8, -2.0)
    for _ in range(3):
        tracker.missed(tracker.objects()[0])  # looked at its place clearly three times: not there
    memory.save(tracker, [], {})
    restored = Tracker()
    memory.restore(restored)
    assert not restored.objects()[0].present


def test_task_scratchpad_state_and_spot_checking():
    pad = TaskScratchpad(goal="find mug")
    pad.mark_visited("kitchen", (2.0, 1.0))
    pad.mark_visited("kitchen", (2.5, 1.2))
    pad.inspected_ids.add(42)
    pad.blockers.append("closed door")
    pad.steps += 3
    for i in range(15):
        pad.record_step(f"  step {i}  ")
    pad.record_step("   ")

    assert len(pad.completed_steps) == 12
    assert pad.completed_steps[0] == "step 3"
    assert pad.completed_steps[-1] == "step 14"
    assert pad.is_spot_checked((2.1, 1.1), radius=0.6)
    assert not pad.is_spot_checked((5.0, 5.0), radius=0.6)
    assert "kitchen" in pad.visited_rooms
    assert "find mug" in pad.summary()
    assert "closed door" in pad.summary()
    assert "Completed: [" in pad.summary()

    pad.reset(goal="inspect table")
    assert pad.goal == "inspect table"
    assert len(pad.visited_rooms) == 0
    assert len(pad.checked_spots) == 0
    assert len(pad.blockers) == 0
    assert len(pad.completed_steps) == 0
    assert pad.steps == 0


def test_visual_ring_buffer_expiry_and_lookup():
    buf = VisualRingBuffer(maxlen=16, max_age_s=15.0)
    img = np.zeros((40, 40, 3), dtype=np.uint8)

    buf.push("front", (0.0, 0.0, 0.0), img, track_ids=(1,), timestamp=100.0)
    buf.push("front", (1.0, 0.0, 0.0), img, track_ids=(2,), timestamp=105.0)
    buf.push("front", (2.0, 0.0, 0.0), img, track_ids=(3,), timestamp=110.0)

    # Frame at 100.0 is 20s older than 120.0 -> should be pruned on push at 120.0
    buf.push("front", (3.0, 0.0, 0.0), img, track_ids=(4,), timestamp=120.0)
    assert len(buf._frames) == 3
    assert {f.timestamp for f in buf._frames} == {105.0, 110.0, 120.0}

    # Query recent frames within 6 seconds of now=120.0 (should find 120.0)
    recent = buf.recent(seconds=6.0, now=120.0)
    assert len(recent) == 1
    assert recent[0].timestamp == 120.0

    # Query frame 10 seconds ago from 120.0 (target=110.0)
    frame = buf.frame_at(seconds_ago=10.0, now=120.0)
    assert frame is not None and frame.timestamp == 110.0


def test_embed_text_semantic_matching_and_query_facts(tmp_path):
    # Synonyms map to identical canonical stems so "spare key" vs "backup keys" match closely
    v_spare = embed_text("spare key")
    v_backup = embed_text("backup keys")
    sim = cosine_similarity(v_spare, v_backup)
    assert sim > 0.98, f"Expected high similarity for synonyms, got {sim}"

    # Empty text returns zero vector and safe cosine similarity 0.0
    v_empty = embed_text("")
    assert np.all(v_empty == 0.0)
    assert cosine_similarity(v_empty, v_spare) == 0.0
    assert cosine_similarity(None, v_spare) == 0.0

    # Semantic facts storage and vector search
    mem = Memory(tmp_path / "facts.db")
    mem.remember_fact("the spare key is under the blue couch", room="living room")
    mem.remember_fact("red tea mug is on the kitchen desk", room="kitchen")

    # Re-inserting exact duplicate should be ignored
    mem.remember_fact("the spare key is under the blue couch", room="living room")
    assert len(mem.notes()) == 2

    # Query using synonym "backup keys under sofa" should retrieve the spare key fact first
    matches = mem.query_facts("backup keys under sofa", top_k=2, min_score=0.25)
    assert len(matches) >= 1
    assert matches[0] == "the spare key is under the blue couch"


def test_volatility_half_life_decay_and_circulate_eviction(tmp_path):
    # Freshness calculation
    assert freshness(1000.0, "blocker", now_wall=1000.0) == 1.0
    assert freshness(1000.0, "blocker", now_wall=1000.0 + HALF_LIFE["blocker"]) == 0.5
    # After ~3.32 half-lives, freshness falls below 0.10
    t_evict_blocker = 1000.0 + 3.4 * HALF_LIFE["blocker"]
    assert freshness(1000.0, "blocker", now_wall=t_evict_blocker) < 0.10

    mem = Memory(tmp_path / "circulate.db")
    tracker = Tracker()
    _seen(tracker, "keys", 2.0, 3.0)  # movable
    _seen(tracker, "table", 4.0, 4.0)  # anchor
    mem.save(tracker, notes=[], now_wall=1000.0)
    mem.record_route_note("kitchen", "cable on floor", 1.0, 1.0, kind="blocker", now_wall=1000.0)

    # At t=1000.0, everything is fresh
    assert len(mem.route_notes(now_wall=1000.0)) == 1

    # At t=6500.0 (~5500s later), blocker freshness < 0.10, but movable (~7200s half-life) is alive
    evicted = mem.circulate(now_wall=6500.0)
    assert evicted == 1
    assert len(mem.route_notes(now_wall=6500.0)) == 0

    # At t=35000.0 (~9 hours later), movable item freshness < 0.10 -> evicted, but anchor persists
    evicted_movable = mem.circulate(now_wall=35000.0)
    assert evicted_movable == 1

    # Verify that the anchor table persists in DB
    restored = Tracker()
    mem.restore(restored, now_wall=35000.0)
    labels = [t.label for t in restored.objects()]
    assert "table" in labels
    assert "keys" not in labels


def test_habit_ema_shifting_across_furniture_anchors(tmp_path):
    mem = Memory(tmp_path / "habits.db")

    # Record keys on table twice
    mem.record_habit("keys", "table", alpha=0.3)
    mem.record_habit("keys", "table", alpha=0.3)
    habits = mem.usual_anchors("keys")
    assert habits[0][0] == "table"

    # Now keys start appearing repeatedly on the desk
    for _ in range(5):
        mem.record_habit("keys", "desk", alpha=0.3)

    habits_after = mem.usual_anchors("keys")
    # Desk should now be the top anchor spot
    assert habits_after[0][0] == "desk"
    assert habits_after[0][1] > habits_after[1][1]


def test_observe_clear_space_overwrites_memory(tmp_path):
    mem = Memory(tmp_path / "clear.db")
    tracker = Tracker()
    _seen(tracker, "keys", 1.0, 2.0)
    _seen(tracker, "backpack", 8.0, 8.0)
    mem.save(tracker)
    mem.record_route_note("hallway", "box in passage", 1.0, 2.05, kind="blocker")

    # Live camera view sees clear space around (1.0, 2.0) with radius 0.6
    mem.observe_clear_space(1.0, 2.0, radius=0.6)

    # Route note was cleared
    assert len(mem.route_notes()) == 0

    # Keys at (1.0, 2.0) was marked present=0
    restored = Tracker()
    mem.restore(restored)
    keys_track = next(t for t in restored.objects() if t.label == "keys")
    assert not keys_track.present

    # Far object at (8.0, 8.0) was not affected
    backpack_track = next(t for t in restored.objects() if t.label == "backpack")
    assert backpack_track.present


def test_person_profile_voiceprint_identification_and_look_update(tmp_path):
    mem = Memory(tmp_path / "people.db")

    rng = np.random.default_rng(42)
    v_alice = rng.standard_normal(64).astype(np.float32)
    v_alice = v_alice / float(np.linalg.norm(v_alice))
    l_day1 = np.array([0.8, 0.2, 0.0], dtype=np.float32)

    profile = mem.upsert_person("Alice", voice_vec=v_alice, look_vec=l_day1, room="living room", now_wall=100.0)
    assert profile.name == "Alice"
    assert isinstance(profile, PersonProfile)

    # Alice speaks again with slightly noisy voiceprint (high cosine similarity) and new outfit
    v_noisy = v_alice + 0.05 * rng.standard_normal(64).astype(np.float32)
    v_noisy = v_noisy / float(np.linalg.norm(v_noisy))
    l_day2 = np.array([0.1, 0.9, 0.0], dtype=np.float32)

    matched = mem.identify_speaker(v_noisy, look_vec=l_day2, threshold=0.70, now_wall=200.0)
    assert matched is not None
    assert matched.name == "Alice"
    assert np.allclose(matched.look_vec, l_day2)

    # Verify look_vec was updated in DB
    rechecked = mem.identify_speaker(v_noisy, threshold=0.70)
    assert rechecked is not None
    assert np.allclose(rechecked.look_vec, l_day2)
