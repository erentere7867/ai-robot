# Evaluation

Three instruments, from fast to thorough. Use the one that decides the question at hand.

| Tool | What it measures | Time |
|---|---|---|
| `python -m pytest -q` | the harness without a model: control, perception, people, memory, guards | < 1 min |
| `scripts/eval_decisions.py` | the model's first tool call in ~20 situations (one call each) | ~1 min |
| `scripts/eval_scenarios.py` | whole tasks in the MuJoCo simulator, graded from ground truth | 5-30 min |

Plus the timing probes: `scripts/probe_latency.py` (where inference time goes, from the server's own timings),
`scripts/bench_control.py` (control-loop lateness under load) and `scripts/bench_localization.py`.

## Scenarios

`scripts/eval_scenarios.py` sets up a room, sends utterances through `RobotHarness.submit_message` exactly as
the console or dashboard would (in the simulator that includes the person's voice reaching the microphone
array), and grades what happened from **simulator ground truth**: the true poses, the true object positions,
and what the cameras could actually see. The robot gets none of that; it only has its sensors and tools.

```bash
python scripts/eval_scenarios.py --list
python scripts/eval_scenarios.py --scenarios people --seeds 2 --no-server-start   # a set, two room layouts
python scripts/eval_scenarios.py --scenarios come_here,follow_me --seeds 3
python scripts/eval_scenarios.py --backend null --scenarios greet                 # runner smoke test
```

| Set | What it covers |
|---|---|
| `core` | greet, describe, turns, move, go to / behind / between, find something behind, dock, come back, look around, memory across a restart |
| `people` | "come here" (person ahead, behind, to the side), "look at me", "follow me" (person walking), "do a full circle around the box behind you", "what's behind you?" |
| `realtime` | follow a rolling ball, stop while following, stop mid-way when interrupted, go to a ball that rolls away |
| `tasks` | several-part requests in a person's own words ("go to that ball and come back", circling unseen or clockwise) |
| `hard` | repeats, chained steps, references to earlier turns |
| `paraphrase` | held-out rewordings of the other sets; fix failures with general mechanisms, never word rules |
| `session` | requests that failed in real sessions |

Each scenario gets a fresh harness and memory, autonomy and recording off; each seed jitters the layout
deterministically. The person is in the room only in scenarios that place them, standing and facing the robot.
Graders see through the same three-camera rig as the robot (`visible_object_ids` uses the simulator's own
ground-truth projection, tested to agree exactly). With cameras all around, "what's behind you?" passes when the
answer names what is behind; turning to look is allowed but not required.

The JSON report (default `artifacts/eval/`) is rewritten after each scenario: a summary plus, per scenario and
seed, the grader's reason and details, start and end object positions, and a transcript with every tool call,
reflex, true pose and down-sampled trajectory.

To add a scenario, append a `Scenario(...)` to a set in `robot/eval/scenarios.py` with a grader that reads only
the `ScenarioRun`, and a synthetic test for the grader in `tests/test_eval.py`.

## Current results

Laptop: LFM3-VL-2B Q4_K_M with the F16 vision projector on a GTX 1650 (llama.cpp CUDA), detector on the same GPU
through Vulkan, three simulated cameras, simulated lidar and microphone array.

Two room layouts per scenario (2026-11-03). The last full pass ran before the final fixes for "follow me",
"look at me", "follow the ball" and camera seams, which were verified with scripted runs only:

| Set | Result | Notes |
|---|---|---|
| core | 28/28 | |
| tasks | 13/14 | "find the duck and park next to it": the model parked next to the dock ("it" misread) |
| people | 12/14 then 9/14 mid-fix | come here (front, behind, side) and circling the box behind pass on both layouts |
| realtime | 4/8 then 6/8 | "follow the ball": the model claimed to follow after a search (now caught); interrupt timing fixed |

"Follow me" with the final tracking code: 4/4 layouts followed a walking person at 0.6-1.0 m with one track
throughout (scripted model). Rerun `--scenarios people,realtime,tasks,core` to refresh these numbers.
