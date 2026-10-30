# Capability evaluation

`scripts/eval_scenarios.py` checks whether the robot can carry out general spoken tasks in the MuJoCo
sandbox. Each scenario sets up a room, sends one or more utterances through
`RobotHarness.submit_message`, and grades what happened using **simulator ground truth**: the true
pose, the true object positions and what the head camera could actually see. The robot gets none of
that. It only has its normal sensors and tools.

```powershell
# list scenarios and their utterances
python scripts/eval_scenarios.py --list

# full suite against the running llama-server, 3 scene seeds
python scripts/eval_scenarios.py --backend llamacpp --no-server-start --seeds 3

# a few scenarios, custom report path
python scripts/eval_scenarios.py --scenarios go_to_ball,behind_box --out artifacts/eval/nav.json

# offline smoke test of the runner itself (the null backend is not expected to pass much)
python scripts/eval_scenarios.py --backend null --scenarios greet,memory_name
```

| Option | Meaning |
| --- | --- |
| `--scenarios` | `all` (default) or comma-separated names |
| `--seeds N`, `--seed-start S` | run seeds `S..S+N-1`; each seed jitters object placement deterministically |
| `--backend` | `auto`, `llamacpp` or `null` (default: the config's `llm.backend`) |
| `--no-server-start` | never auto-launch llama-server; use one that is already running |
| `--timeout` | per-utterance timeout in seconds (default 300) |
| `--out` | JSON report (default `artifacts/eval/eval-<time>-<backend>.json`) |
| `--strict` | exit 1 if any scenario fails (for CI) |

Every scenario gets a fresh harness and a fresh temporary memory DB, with autonomy and event recording
off. For the memory scenarios, the harness is stopped and restarted on the same DB between the two
utterances. If a scenario throws an exception or times out, it is recorded as a failure with its reason,
and the run continues with the next scenario.

## Scenarios

| Name | Utterance(s) | Pass criteria (ground truth) |
| --- | --- | --- |
| greet | "hi!" | non-empty reply, moved < 30 mm, turned < 10 deg |
| describe | "what do you see?" | names >= 2 object families that were in the camera view during the turn |
| turn_around | "turn around" | heading change 180 +/- 30 deg |
| look_left | "look to your left" | heading change +90 +/- 35 deg (+ is CCW/left) |
| move_forward | "move forward a little" | 80..700 mm along the start heading, heading change < 15 deg |
| go_to_ball | "go to the red ball" | ends <= 450 mm from the ball, ball within +/-40 deg of heading |
| behind_box | "what do you see?", "go behind that box" | box lies between start and end (angle between box-start and end-box < 55 deg), end <= 800 mm from box, no footprint contact or obstacle reflex at the box |
| between | "go between the red ball and the blue cube" | ends <= 300 mm from their midpoint |
| find_hidden | "find the yellow duck" (duck placed behind the robot) | ends facing the duck (+/-40 deg), or names it with a location cue after it was actually in view |
| go_to_dock | "go to your charging dock" | ends <= 550 mm from the dock |
| come_back | "go to the red ball", "now go back to where you started" | left the start by >= 250 mm, then ended <= 300 mm from it |
| look_around_report | "look around and tell me everything in the room" | names >= 4 object families present in the room |
| memory_name | "my name is Deniz" / restart / "what's my name?" | reply contains "Deniz" |
| memory_pref | "remember that my favorite color is green" / restart / "what's my favorite color?" | reply contains "green" (the green mug is removed so it cannot leak from the scene) |

Object families and synonyms: box/carton/cardboard, ball/sphere, mug/cup/cylinder, cube/block, duck,
dock/charger/station. Plurals count. The camera ground truth in `robot/eval/scenarios.py` mirrors the
sandbox detector: frustum, tilt, range and occlusion all match. It agreed with
`SimulationSandbox.detect_objects` on 3999 of 4000 random poses, and a test keeps that agreement above
98% so the graders cannot quietly drift if the simulator's camera changes.

## Report

The JSON report is rewritten after each scenario, so an interrupted run still leaves its partial
results. It contains a `summary` (pass rate, per-scenario counts, model calls, time) and one entry per
scenario and seed. Each entry holds the grader's `reason` and `details`, the object positions at the start
and end, and a `transcript`. Every transcript turn has the utterance, the **last** `robot.response` `say`
(earlier responses can be intermediate), action results with arguments, reflexes, the true pose before
and after, and a down-sampled true trajectory `[x, y, heading, tilt]`.

`model calls` counts `cognition.model_call` events. If the cognition layer does not emit them, the count
is 0. `actions` is completed/failed.

## Code layout

- `robot/eval/scenarios.py`: pure scenario definitions, scene layout, camera and collision geometry,
  and graders. No harness imports. Unit-tested in `tests/test_eval_scenarios.py`.
- `robot/eval/runner.py`: drives a `RobotHarness` through scenarios using only its public API
  (`submit_message`, `arbiter.wait_until_done`, bus subscriptions, `body.true_pose()`/`pose()`, and
  `sandbox.list_objects/move_object/remove_object/reset_default_scene`).
- `scripts/eval_scenarios.py`: the CLI.

To add a scenario, append a `Scenario(name, description, utterances, grader, SceneSetup(...))` to
`SCENARIO_LIST`. Give it a grader that reads only the `ScenarioRun`, and add a synthetic test for it.
