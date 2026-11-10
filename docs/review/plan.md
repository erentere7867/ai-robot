# Sprocket: the proposed plan, for review

10 November 2026. Written by Claude, the coding agent building the robot, for an independent review. It merges three sources: Fable's first architecture report (the project's spec), Fable's second report "360° robot software architecture" (10 November 2026), and what this project has built and measured so far. Everything marked *measured* was run in this project. Everything marked *vendor* or *paper* is someone else's claim. Everything marked *estimate* is a guess.

## 1. What the owner wants

- **A small wheeled indoor robot that does what it is told.** It takes any spoken instruction and carries it out, thinking and acting at the same time, in real time, entirely on the robot. The owner's examples: "come here", "do three circles around the box that's behind you", "follow me", "search for a yellow box in this room; if you find it, come back and tell me", and later "where did I leave my keys?".
- **No behaviour scripted per command.** The owner's words: "I don't want these abilities to be predefined things". On 10 November the owner asked for the best middle ground between Fable's skill-based plans and fully general programs written by the model.
- **Capability over reliability, and it should impress people.** Several models are fine when each earns its place. Neural networks where they genuinely help.
- **On-device first.** A cloud model is optional, used only when Wi-Fi is there.
- **One professional design, decided on evidence.** The owner is not a robotics specialist and has been frustrated by designs that changed every few messages. A choice should change only when a measurement or clear evidence forces it.

Owner decisions so far:
- The board is fixed: Arduino VENTUNO Q, 16 GB.
- 360° camera coverage.
- A preference for the LFM3 family, now reopened only by measurement.
- Fine-tuning runs on Kaggle.
- Still unknown: personal project or product. This decides whether copyleft licences matter.

## 2. Hardware

| Part | Choice | Status |
| --- | --- | --- |
| Brain | Arduino VENTUNO Q: Qualcomm Dragonwing IQ-8275 (8-core Kryo, Adreno 623, Hexagon NPU up to 40 TOPS), 16 GB LPDDR5 shared, STM32H5 microcontroller on an RPC bridge | Fixed |
| Lidar | 2D, Slamtec RPLIDAR C1 (360°, 12 m, 10 Hz) | Default part |
| Front camera | RGB-D (RealSense D435i or Orbbec Gemini 335) | Default part |
| Rear cameras | 2 × wide-angle RGB on MIPI-CSI, facing ±120°, no depth | Default part |
| Other | Wheel encoders, BNO085 IMU, 4-mic array with echo cancellation, bumper, hardware e-stop, 4S LiFePO4 | Default parts |

What is known about the NPU:
- **One usable unit.** *Measured by a third party* ([vq-fast-lfm](https://github.com/heath0xFF/vq-fast-lfm)) on a VENTUNO Q: the board exposes one usable Hexagon unit. Text generation is bound by memory bandwidth, about 45 GB/s divided by the model's size. Q4_0 is the only format at full speed. LFM2.5-2.6B wrote 22.5 tokens/s.
- **Gemma 4 E2B:** about 25–28 tokens/s on this chip, from Qualcomm's own guide, text-only.
- **Several models can share the unit.** llama.cpp's Hexagon backend (experimental) gives each "virtual session" about 3.5 GB of address space, and several sessions can share the one physical NPU.
- **Estimate:** a 4B model at Q4_0 (about 2.5 GB) writes roughly 18 tokens/s, so a 90-token plan takes about 5 s.

## 3. What exists today

Everything so far runs on a laptop (GTX 1650 4 GB, WSL Ubuntu 26.04, ROS 2 Lyrical). The board has not been brought up yet.

### Built
- **Robot API** (Python): about 25 primitives in four groups.
  - Motion: `move`, `turn`, `drive`, `face`, `go_to`, `follow(path)`, `stop`.
  - Perception: `detect`, `find`, `watch`, `look`, `ask`, `scan`, `unseen`.
  - Speech and memory: `say`, `listen`, `remember`, `recall`.
  - Plus `geom` helpers (`dist`, `bearing`, `circle`, `line`).
  - Units are metres, seconds and degrees, with + to the left. The command card the planner reads is generated from the code.
- **Gate** that every program must pass:
  - Python AST whitelist, with "did you mean" hints.
  - Every loop must call the robot, so it can be pre-empted.
  - A dry run in a 2D simulator, with a budget of 2,000 calls or 120 simulated seconds.
- **Runner:** the program runs in a sandboxed separate process, and every robot call is a checkpoint. A stop halts motion in ≤ 300 ms (*measured*).
- **2D simulator:**
  - occupancy grid, A* planning and a collision monitor;
  - 360° perception from the scene's ground truth, with line of sight;
  - object memory with "last seen", camera coverage, unsearched viewpoints, scripted user replies and walking people.
- **Test suite:** 50 instructions in 7 categories (relative motion, object-relative, search, conditional, following, memory, ambiguous or impossible), each with an automatic check.
  - The checks were audited: an idle robot fails all 50. Three lenient checks were found and fixed.
  - 10 hand-written programs pass, and 20 red-team programs are all rejected by the gate.
- **ROS 2 backend:** the same API, with motion through Nav2 actions, pose from TF and the real `/scan`. Perception still comes from the scene's ground truth, because phase 3 is not built.
  - Any suite scene exports to a Gazebo world plus a Nav2 map, and runs on Nav2's TurtleBot 4 simulation.
  - *Measured:* all 9 static hand-written programs pass in both Gazebo and the 2D world.
  - This found two bugs that would also have hit the real robot. After a `drive` there was no stop, so Nav2's velocity smoother coasted the robot about 25 cm further. And Nav2's spin came up about 12° short per 360° because the wheels slip; turns now close the loop on the map's heading.
- **Planner:** LFM3-VL-2B behind llama-server.
  - A grammar fixes the answer form: `say:` line, `plan:` line, then a Python code block. The code inside the block is not constrained.
  - 5 style examples, none of them a suite task.
  - Repairs: up to 2 after a gate rejection, up to 3 in all.

### Measured results

**Photo check M0** (50 COCO photos, human-drawn boxes, Q4_0 QAD with F16 vision projector):

| Test | Q4_0 QAD | Q8_0 base |
| --- | --- | --- |
| Is it in the picture? (yes/no) | 98.3% | 95.7% |
| Which numbered box is it? | 80.7% (asking yes/no first) | 77.3% |

**Planner on the 50-task suite** (laptop GPU, temperature 0, pass = the check holds and ≤ 2 repairs):

| Configuration | Passed | Median first plan | Tokens written |
| --- | --- | --- | --- |
| LFM3-VL-2B Q4_0 QAD, API in radians | 20/50 (40%) | 1.8–1.9 s | 89–101 |
| LFM3-VL-2B Q8_0 base | 14/50 (28%) | 2.4 s | 102 |
| Q4_0 QAD, API in degrees, with `move(metres)` | 19/50 (38%) | 2.0 s | 90 |
| Q4_0 QAD, thinking on (up to 3,000 characters) | 9/50 (18%) | 24.6 s | 412 |

**What the failures were** (Q4_0 QAD, about 30 failures per run):
- **About 24 were valid programs that ran and did the wrong thing.**
  - `drive(1.5, 0, 1)` for "drive 1.5 m".
  - `turn(-90)` for "turn left", even though the card says + is left.
  - Only checking memory for "is there a backpack anywhere?" instead of searching unseen area.
  - Answering "come here" by saying which room it is in.
- **5–8 used names that do not exist:** `user`, `pi`, `robot.circle`.
- **Repairs rarely helped:** they fixed 1–2 of about 12 failing programs, and the model often wrote the same mistake again.
- **Changing the API from radians to degrees did not change the score,** because the model mostly ignored the new calls.
- **Thinking made it worse and 12 times slower.** Note that PLAN-Bench reports thinking helping multi-step planning by about 31 points on average. Our test used non-thinking few-shot examples and an untrained model, so this may not be the last word.

## 4. Proposed architecture

The robot has three layers at three speeds, one memory they all read and write, and a planner that runs once per task and then steps aside.

### 4.1 Layers
- **L0, microcontroller (1 kHz):** motor control, odometry with IMU, bumper and e-stop, an obstacle stop, and a watchdog on the main computer.
- **L1, ROS 2 Lyrical (10–30 Hz):**
  - slam_toolbox (map once, then localise);
  - robot_localization EKF (wheels and IMU);
  - Nav2: MPPI controller, Collision Monitor, Following Server, Docking Server, keep-out zones;
  - perception and tracking.
- **L2, cognition (seconds):** planner, gate, runner, memory queries, speech.

### 4.2 Perception (adopted from Fable's 360° report)
- **Detector:** YOLOE-26 s or m, with masks, on all three cameras. A base vocabulary of about 1,200 household classes (Ultralytics measured about 19% extra time going from 80 to 1,203 classes), plus task prompts at run time. Colour and size come from the mask and the 3D extent, not from the detector. The challenger is OmDet-Turbo (Apache-2.0, boxes only), tested in phase 3. WeDetect was dropped as default: Chinese class names, no licence file in the repository (checked 10 November), no NPU export.
- **3D:** the front camera's depth. For the rear cameras, Depth Anything 3 Small, scaled by the lidar points that land in the image. Each camera is calibrated to the lidar once.
- **Tracking:** one tracker in the map frame across all cameras (BoT-SORT-style association, using odometry for motion), so an object keeps its ID from camera to camera.
- **People:** a small re-identification embedding (MobileNetV3-Small class), and a gallery of the follow target collected at lock-on.
- **Output:** one stream of tracks, each with ID, label, confidence, map position, size, colour, last-seen time and an embedding pointer. The rest of the stack reads only this.

### 4.3 Localisation and places (adopted)
- The robot maps once, saves the map and localises at boot from the dock's known pose. Global relocalisation is the fallback.
- Named places come from a guided tour ("this is the kitchen"), plus places saved during tasks.
- The grid is segmented into rooms.

### 4.4 Memory (adopted, following eMEM, RAVEN and DualMap)
- **Storage, following eMEM:** one in-process database. SQLite holds records and links, an HNSW index holds embeddings, an R-tree holds positions.
- **What gets stored:**
  - objects as entities with sighting history;
  - keyframe image embeddings (SigLIP class) every few seconds and on scene change;
  - task events;
  - facts the user told it, tagged as told.
- **Keeping it true:**
  - furniture is a fixed anchor, and small things are movable items attached to anchors;
  - a clear view that does not see a thing is negative evidence;
  - confidence decays with time since the last sighting;
  - duplicates are merged at the dock.
- **Recall calls:** `locate`, `last_seen`, `near`, `recall`, `remember`. Each returns a few lines of text. Retrieval is an embedding match with place and time filters, and no language model re-reads the results.

### 4.5 Commands: one manifest
| Kind | Calls |
| --- | --- |
| Skills: deterministic, tested, recover locally | `go_to(place, pose or object)`, `approach(target, distance)`, `search(description, area)`, `orbit(target, laps)`, `follow(person)`, `dock()`, `wait(condition or seconds)`, `save_place(name)` |
| Primitives | `move(metres)`, `turn(degrees)`, `drive(v, w, t)`, `face(target)`, `stop()`, `pose()`, `scan()` |
| Senses | `detect`, `watch`, `check(question, target)` (yes/no with a probability), `describe(target)` |
| Memory | `locate`, `last_seen`, `near`, `recall`, `remember` |
| Speech | `say`, `ask(question)` (speaks, then returns the answer) |

- **One file generates four things,** so they cannot drift: the planner's command card, the decoding grammar, the gate's whitelist and the fine-tuning data.
- **Skills follow Fable's specifications:**
  - `search` asks memory, then the thing's usual anchors, then explores unseen area ranked by likelihood (OpenFrontier and VLFM as references), and confirms with `check` before reporting.
  - `orbit` counts laps by accumulated angle round the target.
  - `follow` uses Nav2's Following Server fed by person tracks with re-identification.
- **All arguments are bounded,** and every motion goes through Nav2's Collision Monitor and the microcontroller's obstacle stop.

### 4.6 Planning: the middle ground
- **Programs call skills as well as primitives.** The model writes a short Python program against the manifest. Most tasks become two or three skill calls. New tasks drop down to primitives and loops.
- **Fable's worked example as a program:**
  ```python
  start = robot.pose()
  try:
      box = robot.search("blue box", "this room")
      if box:
          robot.orbit(box, laps=3)
  finally:
      robot.go_to(start)
      robot.say("I found the blue box and circled it." if box else "I couldn't find a blue box.")
  ```
- **The gate is the validator, the runner is the executor.**
  - The gate checks the whitelist, names defined before use, preemptible loops, and a dry run in the 2D world.
  - The runner gives every call a timeout. Skills retry locally first, "stop" pre-empts at once, and `finally` always runs.
  - The model is called again only when a step fails after its own retries, with a one-line summary.
- **What the planner sees:** the command, the current place, the battery, and a few lines from memory lookups (or a short table of nearby known objects). No images, no map.
- **Escalation:** two failures go to a stronger tier (the cloud when online), then to a spoken question. Never a guess.
- **Open option, not decided (the owner's question):** a fast lane where a simple command gets one call, such as `do: robot.follow(robot.user())`, with the call's log-probability as its confidence and "ask" when it is low. Measure before adopting.
- **Possible improvement to measure:** constrain call names in the grammar to the manifest. Our data says this fixes at most 5–8 of about 30 failures, because most failures are valid calls with the wrong meaning.

### 4.7 Models

| Job | Candidate(s) | Status |
| --- | --- | --- |
| Planner | LFM3-1.7B (text), LFM3-VL-2B, LFM3-Omni-2B, Qwen3.5-4B, Gemma 4 E4B | Bake-off on the 50-task suite with the final manifest, then fine-tune the winner |
| `check` (yes/no with probability) | d1-2B (Liquid's calibrated decision model on LFM3-VL-2B) | To measure on the M0 photos |
| `describe` | LFM3-VL-2B, or the planner if it is a vision model | — |
| Detector | YOLOE-26 s/m against OmDet-Turbo | Phase 3 |
| Depth for the rear cameras | Depth Anything 3 Small, scaled by lidar | Phase 3 |
| Re-identification | MobileNetV3-Small-class embedding | Phase 3 |
| Memory embeddings | SigLIP class (MobileCLIP2 as alternative) | Phase 3 |
| Speech | LFM3-Omni-2B (one model hears and speaks) against Moonshine or Whisper plus Kokoro or Piper | Phase 6 |
| Learned navigator (later, as a tool) | LightNav-0, InternVLA-N1, or Qwen-RobotNav if its weights appear | After phase 3 |

Why a bake-off rather than a fixed pick:
- **Size helps planning most up to 4B.** PLAN-Bench (28 VLMs, embodied planning) measured Qwen3.5 at 20.2% for 2B, 43.2% for 4B and 49.2% for 9B.
- **LFM3 was not evaluated there.**
- **LFM3-VL's card says it is "not recommended for coding".**
- **A 4B model fits the board's time budget** (*estimate*: about 5 s per plan).

### 4.8 Training
- **Data:**
  - a large cloud model writes a few thousand instruction-and-program pairs for random simulated rooms;
  - keep only pairs that pass the gate and pass their check in the simulator;
  - add the model's own sampled programs that pass (rejection sampling);
  - the 50-task suite and a new 100-command test set are held out.
- **Stage 1:** SFT (LoRA) on Kaggle (2 × T4 or P100).
- **Stage 2:** GRPO with the simulator checks as the reward. Liquid ships SFT and GRPO notebooks for LFM3.
- **Deploy:** merge, quantise to Q4_0, and run on the NPU.

### 4.9 Simulation
- **The 2D world** is fast. It does the dry run, generates training data and is the main test bed.
- **Gazebo with Nav2** is for integration. The next step there is real furniture models and a detector on the simulated camera.
- **Habitat with scanned homes** is an option for photo-realistic tests later.
- **Detector accuracy is judged on real photos,** never on simulator images.

## 5. Build order from here

| Step | Build | Test |
| --- | --- | --- |
| 1 | The skills (`search`, `orbit`, `approach`, `follow`, `dock`, places, `wait`) in the 2D world and Gazebo | One test per skill, plus the 50-task suite with hand-written programs |
| 2 | Manifest-generated card, grammar and gate. Rerun the suite with untrained models | Pass rate per category |
| 3 | Planner bake-off (5 candidates), then SFT and GRPO on Kaggle | ≥ 80% of the suite with ≤ 2 repairs; the new 100-command set; plan ≤ 8 s median on the board |
| 4 | Board bring-up: models on the NPU, the photo check repeated on the board | ≥ 20 tokens/s; M0 scores hold |
| 5 | Base mobility, mapping, navigation, places, docking | 20 trips between named places without help |
| 6 | Perception, tracking, memory | Positions of 5 objects within 20 cm; 10 objects located after one tour |
| 7 | Speech, follow with re-identification, long-run memory | Spoken tasks in a room not used in development; a week of daily use |

## 6. Where this differs from Fable's 360° report, and why

1. **Plan language: Python programs instead of JSON skill plans compiled to BehaviorTree.CPP.**
   - Fable's format (steps, `if`, `finally`, no loops, no arithmetic, no direct motion) cannot express "rock left and right three times", "spin slowly for five seconds", "drive a square", "drive forward until something is half a metre ahead" or "drive a figure of eight around the two chairs". The owner's goal is any instruction.
   - Python with skills keeps Fable's plans as a subset.
   - The cost: free code is harder for a small model to get right than a constrained list. No direct evidence was found for either format with small models on robot tasks. This is the biggest open question.
2. **Executor: our Python runner instead of BehaviorTree.CPP.** The runner already gives timeouts, pre-emption at every call, `finally` and a dry run. We lose Groot2's live view.
3. **The planner keeps primitives (`move`, `turn`, `drive`).** Fable gives the planner no velocity control. Here, primitives are bounded and safety-filtered like every other call.
4. **Planner tiers: a measured bake-off instead of LFM3-1.7B plus Qwen3.5-9B.** PLAN-Bench shows 9B only about 6 points above 4B, and 9B costs 5.6 GB and roughly half the speed. The "deep tier" is the cloud when online, unless the bake-off says otherwise.
5. **Planner input:** Fable gives the planner memory lookups only. This plan currently also gives a short table of known objects with directions already worked out ("behind-left, 1.4 m"), which the planner can use without making extra calls. Open.

## 7. Risks and what I am least sure of

1. **Whether a 2B–4B model reaches 80%,** even fine-tuned and with skills. The bet is skills plus SFT plus GRPO on simulator rewards.
2. **Python against structured plans for small models** (section 6, point 1).
3. **NPU contention.** A detector on three cameras, depth, embeddings, the planner and d1 all share one Hexagon unit. The runtimes are young: llama.cpp's Hexagon backend is experimental, and running vision encoders on the NPU is unproven on this board.
4. **Sim-to-real.** The skills are being built against perfect ground-truth perception. Real detections miss and mislabel, and the 2D world does not yet model that.
5. **Thinking.** It hurt in our test but helps planning on PLAN-Bench. It may deserve a retest after fine-tuning, though its latency (about 10 times longer) is a problem on the board.
6. **The line between skills and generality.** Too few skills and the model must write fragile code. Too many and the robot becomes a menu. The model may also force a skill onto a task that needs composition.
7. **Licences.** YOLOE-26 is AGPL-3.0. That matters only if this becomes a product.
8. **Vendor numbers.** Most model claims above are the makers' own. Only the measured rows in section 3 are ours.

## 8. Sources
- [LFM3 release](https://huggingface.co/blog/LiquidAI/lfm3)
- [LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B)
- [LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B)
- [PLAN-Bench](https://arxiv.org/html/2610.18836v2)
- [Real-world VLN survey and evaluation (IEEE TASE 2026)](https://arxiv.org/abs/2607.09792): end-to-end camera-only methods went from 61% in simulation to 22% on a real robot; a hierarchical system reached 51% on the real robot.
- [YOLOE / YOLOE-26 docs](https://docs.ultralytics.com/models/yoloe/)
- [WeDetect](https://github.com/WeChatCV/WeDetect)
- [OmDet-Turbo](https://arxiv.org/abs/2403.06892)
- [eMEM](https://www.alphaxiv.org/abs/2606.03374)
- [RAVEN](https://arxiv.org/html/2606.25206v1)
- [DualMap](https://www.alphaxiv.org/abs/2506.01950)
- [DynaMem](https://arxiv.org/html/2411.04999)
- [STaR](https://arxiv.org/html/2602.09255)
- [Qwen-RobotNav](https://huggingface.co/papers/2606.18112)
- [LightNav-0](https://huggingface.co/LightOriginsHQ/LightNav-0)
- [InternVLA-N1](https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN)
- [HoloAgent-0](https://arxiv.org/abs/2606.23565)
- [llama.cpp Snapdragon backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md)
- [GenieX](https://github.com/qualcomm/GenieX)
- [vq-fast-lfm](https://github.com/heath0xFF/vq-fast-lfm)
- [Qualcomm Gemma 4 E2B on IQ8](https://dragonwingdocs.qualcomm.com/blogs/gemma-litert-lm-on-iq8)
- [Nav2 Following Server](https://docs.nav2.org/configuration/packages/configuring-following-server.html)
- [BTGenBot-2](https://arxiv.org/html/2602.01870)
- [Behavior-tree generation with small VLMs](https://arxiv.org/html/2603.06084v1)
