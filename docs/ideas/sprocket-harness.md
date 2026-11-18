# Sprocket Harness: Act Through Tools, See the Result

## Problem Statement
How might we make Sprocket's on-board vision-language model reliably turn any request into the right actions, with a harness built like modern agent harnesses instead of a form it fills in once per step?

## Recommended Direction (built, 18 November 2026)
The model decides and acts through tool calls such as `go_to(target="4")`. The harness around it borrows the patterns coding agents use:

| Pattern | Before | Now |
| --- | --- | --- |
| Results come back | Each action's result was thrown away; every step was a fresh prompt with a harness-written "Completed steps" list | One running conversation per task: the model's calls, then what happened ("Arrived, facing ball 5, 0.6 m away", "Blocked: …") |
| The model's own format | A custom 7-line form (say, chime, posture, action, target, params, subgoal) every step | Native tool calls. LFM3-VL's template uses the `qwen3_coder` XML style; llama-server renders the tool schemas and parses the calls (checked: several calls in one turn, about 0.7 s). Python code blocks were dropped: LFM3-VL's card says it is "not recommended for coding" |
| It actually sees | A regex wrote the answers to "what do you see?" from the detector's list | Camera pictures with the tracker's ids drawn on them come with every request; `look(direction)` brings other cameras. The 2D simulator now draws perspective camera views, so the vision path is tested there too |
| Errors that teach | Errors became "Sorry, I couldn't find what you meant" | Errors say what went wrong and what is possible: `'box' could be any of: box 1 (0.8 m front-left); box 2 (…). Use the id, or ask the user which one` |
| Speech and mood | say, chime and posture required every step | A plain reply is spoken and ends the task; `say` speaks mid-task. Posture and chime come from events |
| To-do list | The harness kept "Completed steps"; a regex guessed when a task was simple | `todo` / `todo_done`, shown back with every update and on the live page |
| Stop hook | — | The first time the model tries to finish after acting, it is shown the request, what it did and any open to-do steps, and asked to check once |
| Loop guards | — | A third identical call in a row is not run; looking again without moving is answered "you already have those pictures" |
| Checks, not rewrites | A distance of 1 m or more was silently changed to 0.5 | Out-of-range or unknown arguments return an error naming the allowed range or values |
| Interruptions | A new instruction cancelled the task and started over | Words spoken during a task interrupt the current step and join the conversation; "stop" and "wait" act at once |

Deleted with it: `robot/agent/planner.py` (the 7-line form, greeting lists, the visual-question regex, the "mug" echo patch), `robot/agent/context.py`, `_is_simple_single_step` and the other regex routing in `agent.py`, "come back" as an instant keyword, the old SFT builder (`robot/train/build.py`, wrong format), the working-memory classes only the old agent used, the dry-run `clone`/call budget, the duplicate Gazebo suite runner in `stage.py`, and the tests for all of these.

## Measured (50-task suite, 2D world, LFM3-VL-2B Q4_0 on the laptop GPU, temperature 0.1)

| Harness | Passed | First model turn (median) |
| --- | --- | --- |
| Old: 7-line Intent loop, with its regex patches | 16/50 (32%) | 0.94 s |
| New: front camera with each request, others on `look` | 22/50 (44%) | 0.70 s |
| New: all three cameras with each request | 24/50 (48%) | 1.20 s |
| New: front camera, plus fresh pictures after every move | 22/50 (44%) | 0.62 s |
| New, with `go_to` split into `go_to` / `go_to_side` / `stop_near`: front camera | 26/50 (52%) | 0.77 s |
| Same, all three cameras | 25/50 (50%) | 1.20 s |
| Same, front camera, plus "search before saying it is not there; go closer to see what is on something" | 28/50 (56%) | 0.64 s |
| Same, plus 768 px pictures, plain id badges, colours in the object list, stop hook on every request | 27/50 (54%) | 0.8–2.8 s (cache hit or miss) |
| Same code, thinking off (`--think off`) | 26/50 (52%) | 1.0 s; 1.1 s per turn, 5.0 s per task |
| Same code, **thinking medium** (512-token cap; the default) | **32/50 (64%)** | 4.8 s; 4.7 s per turn, 16.3 s per task |

Thinking (18 November 2026, same code, back to back): 2D 32/50 against 26/50; 3D apartment 9/10 against 8/10. The gains
are multi-step motion ("back up a metre, then spin twice", "rock left and right three times", "circle the table twice
clockwise", relative 6/8 against 3/8) and conditionals (6/8 against 3/8); with it the model also answered what is on
the shelf by the sofa and went to "where you were before" for "come back". It costs about 3x the time per task on the
laptop GPU, and on the VENTUNO Q (about 22 tokens/s) 512 tokens of thinking could take 20 s or more per turn: fine for
deciding the harness, a reason for post-training (a trained model needs less of it) before the robot runs on the board.

In the 3D apartment (robot/eval/home3d.py, 10 tasks on real rendered pictures): 5/10 at first, 6/10 after fixing the
semantic ids of furniture with doors and adding the search instruction, 7/10 (twice) with the changes below. It
describes what is in view, counts the two bikes, turns and describes what was behind, finds the fridge, picks the
farther beanbag and the stool as closer than the sofa. It still takes the computer monitor for "the TV" and rarely
works out what is on the shelf or the counter: from 0.22 m the cameras never see a counter's surface, only the tall
things on it from a few metres back.

What the 3D apartment taught the harness (18 November 2026):
- **Picture quality decides colour questions.** Asked "which bike is red?", the model's confidence in the right one
  was 0.73 on the old jagged 512 px picture and 1.00 at 768 px. Pictures are now drawn at twice the size and averaged
  down (no jaggies) and sent 768 px wide (about 365 image tokens instead of 150; about 0.4 s more per turn on the
  laptop GPU, so `image_width` is a setting).
- **Coloured id boxes poisoned colour answers.** The model called the red bike "blue", the colour of the box drawn
  round it. Id marks are now plain black badges with white numbers and thin black-and-white outlines.
- **Colours belong in the object list,** as the real tracker gives them (perception/colour.py, now naming a bike by
  its frame when some of its pixels are clearly coloured).
- **Each request carries the last one whole** (its calls and their results; older requests as one line each with
  their actions), and the robot keeps "where you were before" as a place listed with distance and direction.
- **The stop hook runs on every request:** the model said "I'll head back to the starting point" without calling a
  tool; the check now says plainly when no tool was called.
- **`search` does not move for what is already in view**, and names every other known match ("how many bikes?").

Still the model's, not the harness's: after "go to the red bike", "come back" was answered "I'm on my way back to the
red bike" (said, not done) with the previous request, the place "where you were before" and "no tool call at all" in
front of it.

## Failure audit in 3D (18 November 2026)

The 3D apartment is now the test; the 2D suite stays only as a quick smoke test. `robot/eval/home3d.py` has 27
tasks (moving, going to things, seeing and answering, finding, conditions, two requests in a row, the impossible)
whose checks are worked out from whichever layout they run in. They ran in `apt_0` and `v3_sc0_staging_00`, thinking
medium. Every failure was read and labelled harness, body, check or model; the others were fixed and all re-run.

| Run | Passed | apt_0 | v3_sc0_staging_00 |
| --- | --- | --- | --- |
| audit1, before | 41/54 (76%) | 23/27 | 18/27 |
| audit4, first fixes | 44/54 (81%) | 23/27 | 21/27 |
| **audit5, all fixes** | **48/54 (89%)** | 23/27 | 25/27 |

(audit2 and audit3 stopped counting when the model server went down mid-run; not counted.) Model turn: median
3.3–3.9 s on the laptop GPU.

audit1's 13 failures were 7 harness or body, 4 model, 2 my checks:
- **A sofa the tools could not see (4 failures).** 5.5 m away and plain in the picture, but never in the object list:
  the 2D world's 4.5 m "see range" capped detection. Detection now goes by size in the picture only.
- **Stuck in a corner.** After a search left the robot facing a wall beside the TV stand, every route stalled at its
  first waypoint: the path follower only drove forward arcs, with a forced minimum speed. It now turns on the spot to
  face a waypoint more than 45° off (in `robot/body/controller.py`, which the real robot uses too).
- **"Come back here" with nothing pointing back.** The place list reached the model only with pictures, so after two
  `go_to`s it never saw where the request began. Motion results now end with the room and the places, each with
  distance and direction.
- **A place copied as listed was refused** (`go_to("where you were before (0.5 m behind-left)")`): the tool accepts its
  own listing format now.
- Model: going upstairs (twice; the prompt now states "flat floors only" as a fact of the body, and both pass), "no
  bike" without searching, and the table nearest the sofa (only the farthest table was in view; it never looked).
- Checks: "spin around twice" done as once left and once right is twice round; it passes now.

Found by the later runs:
- **Context overflow.** A 16-turn task kept every turn's 512-token thought and outgrew the 8k tokens of a server slot
  ("Sorry, I couldn't finish that"). Only the newest two turns' reasoning is kept now.
- **Corner cutting into furniture.** The follower switched to the next waypoint 28 cm early even when the straight line
  ran into a cabinet's corner, and the collision monitor (which stops 25 cm short) then held it there. It cuts a
  corner only when that line is clear. Driving to every object in both homes without the model: 140 of 142 reached
  (one stall, one with no path).
- **A cut-off object list.** 26 objects detected, 15 listed: a chair in the picture was missing when asked "how many
  chairs?". The rest now follow by id and name ("Farther: 37 chair, 54 plant, ...").
- `go_to("start")` when already there said "Arrived"; it now says it did not move. "where you were before" is
  described and no longer listed as a second name for here; emoji are dropped before speaking; and my "turn around"
  check measured the whole run instead of the second request.

What still fails (audit5, 6 of 54) is the model's, with what it needs in front of it:
- **"Come back" (3).** It goes to a room ("the living room", "the kitchen"), once to a bike while saying "heading back
  to where we started", instead of "where you were before (where the request before began, 5.2 m behind-right)".
  With nobody in the apartment "come back" has no natural meaning ("come to me"): a person in the 3D home comes next.
- "Rock left and right three times": left twice in a row. "The table closest to the sofa": takes the one table in view
  instead of looking near the sofa. "How many chairs?": two, with a third in the picture and in the list.

The three camera variants above ran on identical code; a difference of two cases is within run-to-run noise at this temperature. All cameras cost about 0.5 s more per turn here, and more on the board (three vision-encoder passes).

Two harness lessons from the transcripts:
- **The 2B model fills every optional argument.** With `side` and `stop_distance` on `go_to`, it wrote `side="front"` or `side="ahead"` and `stop_distance=1`–`2` on most calls ("come here" became "go behind the user, 2 m away"); with a `keep` flag on `face`, "turn to face them" never ended. A different behaviour is now a different tool.
- **Restating the task in updates matters.** When camera updates arrived as plain user messages, the model read them as the request again and turned 90 degrees sixteen times. Updates now say what the request was, what is done, and that they are not a new request.

The new harness has no patches tuned to suite cases. What still fails is mostly the untrained 2B model's planning: it picks `drive` for a square, stops after "go to the ball" without coming back, fills optional arguments it was not asked for, and rarely writes a to-do list unprompted. That is what post-training is for.

## Key Assumptions to Validate
- [ ] Post-training on this exact tool format lifts the suite to 80%. → Build SFT data in this format: a large model writes tool-call transcripts in random rooms, keep only those that pass the simulator's checks, train, then reinforce on the checks.
- [ ] A bigger model helps more than more harness. → Rerun the suite with a 4B vision-language model on the same harness, timed.
- [ ] Turns stay at 3 s or less on the VENTUNO Q. → Count tokens per turn on the suite, measure the image prefill on the board.
- [ ] All three cameras with each request beat the front camera plus `look`. → `--cameras all` on the suite.

## Not Doing (and Why)
- Keyword or regex routing of instructions: each patch fixes one test and moves away from "understands anything".
- Python code blocks as the action format: not the model's trained format, and its card advises against coding.
- Re-planning on a timer: the model thinks when something happens.
- Thinking without a cap: uncapped, the model wrote 400 tokens before one tool call. Capped (medium, 512 tokens) it
  is on by default and measured better (above).
- A learned low-level driver: later, as one skill, only if it beats the coded one on the suite.
- Pet behaviours: next, on top of the same tools.

## Open Questions
- Which tools are missing? "Drive around the room close to the walls" and "go behind the box" have no good tool yet.
- Should `go_to` keep the optional stop distance? The model fills it in when nobody asked.
