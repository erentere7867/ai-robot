# Sprocket: Architecture

Combined specification, 10 November 2026. Base: Fable's architecture report. Changes from measurements in this project are marked **Amended**.

## 0. Current design (merged 10 November 2026)

This section overrides the rest of the document where they differ. It merges the original report, Fable's "360° robot software architecture" (10 November 2026) and this project's measurements. The owner asked for the middle ground between Fable's skill plans and programs against primitives.

### Why a middle ground
- **Skills are why plans work.** Measured: the untrained 2B planner wrote programs that ran but did the wrong thing (wrong units and signs, never searching): 38–40%. Fable moves that know-how into deterministic, tested skills, so the model only picks a skill and its arguments.
- **Code is why any task stays possible.** Fable's plan format (skill steps, `if`, `finally`, no loops, no direct motion) cannot express "rock left and right three times", "spin slowly for five seconds", "drive a square" or "drive until something is half a metre ahead". Programs can.
- **Result:** one command set holding skills, primitives, senses and memory, which the planner's programs call.

### Commands (one manifest), as built 11 November 2026 after Fable's review
`robot/agent/api.py` is the manifest: the planner's command card, the gate's whitelist and argument check, and the dry run all come from it. Units are metres, seconds and degrees; directions are words, never signs; unit arguments are keyword-only and named after the unit (`move(metres=1.5, direction="backward")`, `turn(degrees=90, direction="left")`), because the measured failures were `drive(1.5, 0, 1)` for "1.5 m" and `turn(-90)` for "left".

| Kind | Calls |
| --- | --- |
| State | `pose()`, `battery()`, `user()` (the person talking), `room()`, `obj(id)` (a table object as it is now) |
| Motion | `move(metres=, direction=)`, `turn(degrees=, direction=)`, `spin(turns= or seconds=, speed=)`, `face(target)`, `go_to(target, distance=)`, `drive(metres_per_second=, degrees_per_second=, seconds=)`, `drive_path(points)`, `stop()` |
| Skills | `search(description, area=)` (memory, then exploration; a clear look that misses is negative evidence), `orbit(target, laps=, direction=)` (picks a ring free of things and people, or fails with the reason), `follow(person, distance=)` (re-plans a route to the person twice a second), `spot(target, side=)` (a free point behind, in front of, left or right of a thing, as seen now), `wait_for(description, seconds=)`, `wait(seconds)`, `save_place(name)` |
| Senses | `detect(description)` (in view now), `where(description)` (everything seen so far), `check(question, target)` (yes/no), `describe(target, question=)`, `clearance(direction)` |
| Speech and memory | `say`, `ask` (speaks, returns the answer), `listen`, `remember`, `recall` |

A target is a point, a Pose, an Object, or a string naming a saved place, a room or a thing seen. Objects carry `side` (the bearing in words) and `color`. The rule for the line between skills and program (Fable): a call is a skill when it needs feedback faster than the planner; sequencing, counting, conditions and simple geometry stay in the program.

- **Readback (built).** The gate's dry run is turned into sentences ("drive backward 1.0 m; spin left 2.0 turns; end 1.0 m behind the start") by `robot/agent/readback.py`, from the call log and pose trace, so it cannot contradict the code. With `Agent(readback=True)` the planner is asked whether the readback matches the instruction before the robot moves; a "no" becomes a repair.
- **Instant commands (built).** "Stop", "wait" and "come back" are matched by keyword on the transcript (`robot/agent/instant.py`) and never wait for a model.
- **Checks hardened (built).** Besides an idle robot, a set of plausible-but-wrong programs (`robot/eval/wrong.py`) must fail every check they target.
- **Live page (built).** `robot/interfaces/live.py` shows the instruction, the program with the running line highlighted, the readback, what was said and a map with the robot's path. `python -m robot.interfaces.demo` runs the 2D world in real time behind it.
- **Background skills (built).** `follow(person, background=True)` and `face(target, background=True)` keep running while the program waits or asks ("follow me and tell me if you see a cat"), until the next motion or "stop".
- **Detector noise in the 2D world (built).** `Scene.noise` (0 to 1) makes the detector miss things, more often far away, and place them a little off. The robot watches for 2 s before a task, as a running robot would have. All hand-written programs pass at noise 0, 0.5 and 1.
- **Measured, untrained LFM3-VL-2B on the suite:** keyword-only arguments 32% (the model wrote `turn(90, "left")` correctly but positionally, and the gate rejected it); positional arguments accepted 46%. The readback self-check is off: the yes/no answer rejected all 92 programs, and P(yes) separated right from wrong only weakly (AUC 0.61).
- **Training data (built).** `robot/train/` generates random rooms (one or two rooms, household catalogue, random colours), 36 task families across the seven categories with spoken-style phrasings, and teacher-written programs. Every example runs in the 2D world and passes its own check before it is kept; instructions within 0.8 word overlap of a suite instruction are dropped. `robot/train/kaggle_sft.ipynb` fine-tunes LFM3-VL-2B with LoRA on Kaggle (Liquid's recipe) and produces a GGUF adapter for the QAD Q4_0 model and a merged Q4_0 model.

### Planning
- The model answers `say:`, `plan:` and a short Python program that composes skills, primitives and memory calls, with loops and `try`/`finally` where needed.
- **Open option, not decided (owner's question):** a fast lane where a simple command gets one call (`do: robot.follow(robot.user())`), with the call's log-probability as its confidence. Measure before adopting.
- **Gate = validator, runner = executor.** Whitelist from the manifest, names defined before use, every loop preemptible, dry run in the 2D world. Every call has a timeout, skills retry locally first, "stop" pre-empts at once, `finally` always runs. The model runs once. It is called again only when a step fails after its own retries, with a one-line summary.
- **What the planner sees:** the command, current place, battery, and a few lines from memory lookups. No images, no map.
- **Escalation:** two failures go to a stronger tier (the cloud when online, or a larger on-board model if the bake-off justifies one), and then to a spoken question. Never a guess.

### Perception, memory, localisation (Fable's design, adopted)
- **Detector:** YOLOE-26 s/m with masks on all three cameras. A base vocabulary of about 1,200 household classes, plus task prompts. Colour and size come from the mask and the 3D extent. 3D comes from the depth camera, or from the lidar plus Depth Anything 3 Small (scaled by lidar points) where there is no depth. One map-frame tracker across the cameras (BoT-SORT-style association with odometry). People get a re-identification gallery (MobileNetV3-Small-class embeddings).
- **Memory, following eMEM:** SQLite for records, HNSW for embeddings, an R-tree for positions. It holds object entities with sighting history, keyframe embeddings (SigLIP-class) every few seconds, task events and told facts. Furniture is a fixed anchor and small things are movable items attached to anchors (DualMap). A clear view that does not see a thing is negative evidence. Confidence decays with time since a sighting. Duplicates are merged at the dock. Retrieval is an embedding match with place and time filters, with no language model re-reading the results.
- **As built in the 3D simulator (12 November 2026).** `robot/ros/run3d.py` runs the whole stack: Gazebo with Fuel furniture, Nav2, and Sprocket's body (a 640×480 RGB-D front camera at 86°, two 1280×720 rear cameras at ±112° with 155° lenses, an RPLIDAR C1-like lidar at 0.27 m with 500 points per turn, an IMU).
  - *Detection.* YOLOE-26s on every camera at the camera's own resolution (shrinking the rear frames to 640 px dropped a person 4 m away from 0.84 to 0.13 confidence), threshold 0.15, one box per thing (same-name boxes inside each other are one). The vision model names each new object from its crop, choosing from the robot's words (8 of 10 right; answering freely 5 of 10), names it again when it is seen twice as large, and drops it when it sees nothing: propose fast, verify slowly.
  - *Placing.* Each colour frame is paired with the lidar scan taken with it, and with a depth image seen from the same viewpoint: taken within 30 ms, or while the robot moved under 1 cm and half a degree (the depth stream runs slower than the colour one: by the exact stamp, only one frame in four had depth; with this rule 118 of 146 standing, while turning a mismatched image is refused and the lidar serves). Paired with the newest instead, a depth image 200 ms older put one front sighting in five over 30% too far while the robot turned. Depth gives the nearest surface a solid cluster of pixels agrees on (a fixed percentile read the wall behind chair legs); a box whose depth points all lie on the floor is the floor. The rear cameras use the floor estimate (where the box meets the floor), refined by the lidar only within 12% (with a 35% window the beam, passing between chair legs and under table tops, ran 11-18% long) and never for things below the beam; a box that runs off the bottom of the picture gets no floor estimate. Size comes from the angle the box spans. Every place carries its uncertainty.
  - *Tracking.* One map-frame tracker joins detections by place and size, not label, names each track by a score-weighted vote, weighs each sighting by its precision (a scalar Kalman filter, so a still thing settles instead of twitching; people may move faster), joins the part of a thing cut by a picture's edge to the thing, merges same-name twins, and marks a thing absent after three clear looks without it. The vision model's name wins over the detector's in both directions (the detector once called a backpack a person).
  - *People.* Each person carries the colours of their clothes (a hue-saturation histogram of the middle of their box, brightness left out). A person lost behind furniture is known again when they reappear within walking distance and look alike, so "follow me" keeps its person; two people dressed clearly differently do not swap when they pass close; one person seen by two cameras in different light is merged. On the robot, the voice's direction (microphone array) will also mark the user.
  - *Measured.* Living room, robot still: 7 of 9 objects, median error 0.08-0.10 m, no phantoms. Turning slowly: 0.07-0.10 m, objects steady to a few centimetres (before the fixes: 0.20 m, jumps up to 0.4 m, duplicates). Not found: a white floor lamp and a 15 cm plant in far corners.
- **Memory (built, stage A).** `robot/mind/memory.py`: one SQLite file holds the objects (place, colour, sightings, first and last seen, present or gone), the facts the robot was told, and the places it was shown; written every 5 s and at shutdown, read back into the tracker at start. A second session in the same room started knowing six objects, and its new sightings joined them. Embeddings and the R-tree are stage B.
- **Mapping and rooms (built).** `robot/ros/explore.py`: with slam_toolbox, the robot turns once, then drives to frontier cells chosen by the unknown space around them (a frontier's middle was useless: early on it is a ring around the robot) until none is left, and saves the map in the house frame. Two rooms mapped from scratch in 179 s: 96% of the free floor, and perception found the kitchen's objects on the way. Later sessions localise (AMCL) and dry-run programs in that map, with unknown space blocked (`--own-map`). `robot/ros/rooms.py` splits the map at doorways (free floor shrunk by half a door's width falls apart into one core per room) and names each room from the tour ("this is the kitchen" saved inside it), else from its furniture (fridge: kitchen; sofa, TV: living room; bed: bedroom), else "room 1".
- **Speech (built).** `robot/interfaces/voice.py`: Whisper base.en hears, Piper (en_US lessac) speaks. Round trip on the laptop's CPU: 6 of 6 instructions heard word for word, 0.5-0.6 s to hear, under 0.15 s to speak. On the live page, "hold to talk" sends speech to the same inbox as typing (so "stop" acts at once, and an answer goes to the program that asked); "speak aloud" plays what the robot says. On the robot the microphone is always open (`microphone`): the stream is cut into utterances at pauses against a learned noise floor (`Listener`), and only speech meant for the robot is acted on (`addressed`): what follows its name ("Sprocket, go to the chair"), "stop" from anyone, and anything while a program waits for an answer.
- **Odometry (built).** robot_localization's EKF fuses the wheels' forward speed with the gyro's yaw rate and publishes odom → base_link in place of the wheels (sideways speed held at zero; no compass heading indoors). Wheels slip when turning on the spot (a 360° Nav2 spin came up 12° short on wheels alone); with the gyro, the localisation's corrections during a slow turn fell from 2.4-3.2° to 0.4-1.2°.
- **When a task fails (built).** After the on-board planner's repairs, an optional stronger planner (`Agent(escalate=...)`: the cloud tier, or a larger on-board model) gets one full try; then the robot says plainly what went wrong ("I couldn't find what you meant", "I wasn't sure which way to go", "I couldn't find a way there") and asks to be told another way. It never guesses.
- **Full stack, measured 12 November 2026** (Gazebo, Nav2 localising and planning in the robot's own maps, EKF odometry, three cameras, YOLOE-26 and the vision model naming objects, 12 suite cases, each judged against the true room). Hand-written programs: **9 of 12**; the three failures are missing simulator assets, not the robot (no door model for C1, a green cylinder standing in for a mug in S6, plain coloured shapes for C5's boxes). The untrained planner's own programs: **3 of 12** (R7, O1, O6; 4 to 5 of 10 in earlier runs). The gap between 9 and 3 is the planner's program quality, which the fine-tuning on Kaggle is for. Two fixes came out of these runs: a move that ends within 10% (10 cm at most) of its distance is done (a dry run stopped 5 cm short of a box placed 0.3 m too close and failed the whole program), and rooms in the robot's own map take their names from the guided tour.
- **Planner prompt and conversation (12 November 2026).** The rules had taught a bug ("call save_place("start") before leaving", though "start" is saved by itself: the robot saved it again at the chair and "come back" went nowhere) and showed none of the common moves. The rules now name the patterns the model got wrong (directions as words, `spot` for behind or in front, `orbit`, `go_to`'s distance as the stopping distance, search everywhere unless a room is named, `check` for a thing's state, `recall` for told facts, do only what was asked), and the eight worked examples show them. The planner now hears the conversation: the last four instructions, each with what the robot did, objects by id ("drive to the brown table 3"), so "no, the other one", "again" and corrections work. 2D suite: 52% (from 40-46%); the remaining failures are the model writing nonsense it was told not to write, which fine-tuning is for. The training data carries the new rules and two conversation families ("the other one", "again").
- **Objects while driving (12 November 2026).** Measured against Gazebo's true pose on a drive round the room: localisation was 0.10 m and 1.2 degrees off (worst 0.25 m, 6 degrees) with Nav2's default AMCL, and about 22 tracks stood for 9 objects. AMCL now uses 150 of the lidar's 500 beams, updates every 10 cm or 6 degrees and trusts the gyro-fused odometry more: 0.03-0.04 m and 0.4 degrees (worst 0.07 m, 3 degrees). Each sighting's uncertainty includes the robot's pose uncertainty (AMCL's covariance, and 50 ms of timing at the turn rate) at the thing's range, so matching and merging allow for it (capped at 1 m and 0.6 m, or neighbours run together); things of a size merge whatever their names when closer than their sightings' scatter; the vision model first confirms the detector's name with a yes or no and a confirmed name is not asked again; a crop for naming has the smaller things detected in it painted out; a thing too close to fit in the picture is not counted as missing. After a drive: one track per real object, most within 0.1-0.2 m. Still: a sofa confirmed as "chair", a lamp on a stand named "table", and a stray chair track or two.
- **Laptop setup (measured).** Everything heavy runs on the GTX 1650 inside WSL: llama.cpp built with CUDA (plan 2 s with the prompt cached, 3-9 s while Gazebo runs; on the CPU it was 29-58 s, because Gazebo, rendering on the CPU, starved it), YOLOE in full precision (64 ms a 1280-px frame; half precision was 3× slower on this card), and Gazebo's rendering through Mesa's Direct3D 12 driver.
- **Localisation:** slam_toolbox maps once, then localises. robot_localization fuses wheels and IMU. Boot starts from the dock's pose. Named places come from a guided tour. The grid is segmented into rooms.

### Models (settled by measurement)
- **Planner:** a bake-off on the 50-case suite, using the final manifest, between LFM3-1.7B, LFM3-VL-2B, LFM3-Omni-2B, Qwen3.5-4B and Gemma 4 E4B. Fine-tune the winner on Kaggle. First SFT, on plans written by a large cloud model and kept only if the gate accepts them and they pass in the simulator. Then GRPO, with the suite's checks as the reward.
- **`check`:** d1-2B (calibrated yes/no). **`describe`:** LFM3-VL-2B, or the planner itself if it is a vision model.
- **Speech:** LFM3-Omni-2B against Moonshine or Whisper plus Kokoro or Piper.

### Build order from here
1. Skills in the 2D world and Gazebo (`search`, `orbit`, `approach`, `follow`, `dock`, places), each with its own test. **Done** except `dock` (needs the charging contacts).
2. The manifest, the two-lane grammar and parser. Rerun the suite untrained. **Done** (whole programs; the one-call fast lane stays an option to measure).
3. The planner bake-off, then fine-tuning. **The owner kept LFM3-VL-2B; the SFT data and the Kaggle notebook are ready**; GRPO after.
4. Board bring-up, base, mapping, then perception and memory on hardware (phases 0–3). **In simulation: done** (mapping by exploration, localisation in the robot's own map, perception, memory stage A, rooms); hardware next.
5. Speech, then long-run behaviour (consolidation, moved objects). **Speech done** (Whisper, Piper, always-listening ear with addressing); moved objects are handled by negative evidence; memory stage B (embeddings, R-tree) and consolidation remain.

Still open in the simulator: a door model (C1 cannot see whether a door is open), a mug and other small things with real 3D models (S6), and far small things on the 155° rear lenses (a fisheye lens model would give their centre about three times the pixels).

## 1. How to use this report

This report is a build specification for a coding agent. It defines a wheeled robot that takes any spoken or typed instruction, works out its own plan, and executes it in real time with all AI running on the robot.

Read it in order once, then build by the phases in section 13. Each phase ends in acceptance tests; do not start a phase until the previous one passes.

- **Decided, do not change without asking the owner:** the brain board, the three-layer split (reflex / perception / cognition), and the safety rules in section 9. **Amended by the owner on 10 November 2026:** the planner's programs call tested skills as well as primitives (section 0); the original rule of primitives only is replaced.
- **Defaults, swap freely if a measurement says so:** every named AI model, every sensor part number, every rate and threshold marked *target*. Each model sits behind an interface so a swap touches one file.
- **Decided by the owner on 10 November 2026:** LFM3-VL-2B is the planner and visual-question model, and the robot has 360° camera coverage (front depth camera plus two rear cameras).
- **Must be measured on the real board before relying on it:** every latency and memory figure marked *target*. They are engineering estimates from published benchmarks, not measurements of this build.

Keywords: **MUST** is a requirement, **SHOULD** is the default unless there is a measured reason, **MAY** is optional.

## 2. Requirements and design principles

The robot has no task-specific skills. A language model writes a short program for each instruction, and faster layers below it execute that program in real time.

| # | Requirement from the brief | How the design meets it |
| --- | --- | --- |
| R1 | Do any task it is given, with no predefined skills | The planner model writes a Python program per instruction against \~20 primitives (sense, move, remember, speak). No primitive encodes a task. |
| R2 | Understand its surroundings | Lidar SLAM gives geometry, an open-vocabulary detector gives named objects, and an object map remembers what was seen where. |
| R3 | Act in real time | Three layers at three speeds: reflexes at 1 kHz, perception and navigation at 10–30 Hz, thinking in seconds. Motion never waits on the language model. |
| R4 | The brain is the Arduino Ventuno Q | All models run on its 40 TOPS NPU and 16 GB shared RAM. Network is optional. |
| R5 | Control its own wheels for the task | Generated programs command velocities, paths and goals. The microcontroller turns them into wheel speeds. |
| R6 | Go, look, come back and tell | Speech in and out, a remembered user position, and a visual question primitive for checking what it found. |

### Principles

1. **Think slowly, act fast.** The model writes a program once, in seconds. The program then loops at up to 20 Hz on fast perception with no model call in the loop.
2. **Primitives, not skills.** The API exposes what the body can sense and do. "Orbit", "search" and "patrol" MUST NOT exist as functions; the model composes them each time.
3. **Classical where solved, learned where not.** SLAM, path planning and motor control use proven algorithms. Language, open-vocabulary vision and task reasoning use models.
4. **Safety sits below the AI.** No generated program can reach the motors except through a governor it cannot change.
5. **Every model is replaceable.** Each sits behind one interface and is chosen by a measured bake-off, because edge models change every few months.

### Why not one end-to-end model

An end-to-end vision-language-action model was considered and rejected for this robot. Current open ones target arms and need per-robot training data. Navigation foundation models such as [NavFoM](https://arxiv.org/abs/2509.12129) cover search and instruction following, but nothing shows them running on this class of NPU, and none composes "three laps, then report".

The design instead follows the split Google DeepMind uses in [Gemini Robotics 1.5](https://deepmind.google/discover/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/): a reasoning model plans and calls tools, and a lower layer acts. Here the tools are the primitive API, and the plan is code, as in [Code as Policies](https://arxiv.org/abs/2209.07753).

## 3. System overview

The robot is three layers stacked by speed: cognition decides in seconds, perception and navigation run at 10–30 Hz, and the reflex layer runs at 1 kHz on the microcontroller.

Each layer only talks to its neighbours. Commands go down, and state and events come up. A layer keeps working at its own rate when the layer above is busy or has crashed.

&#91;embedded content: architecture · three layers on one board\]

Only the planner model thinks in seconds. Everything below it keeps running without waiting for it.

| Layer | Runs on | Rate | Owns |
| --- | --- | --- | --- |
| L2 Cognition | Linux, NPU | 2–10 s per decision | Speech, planner model, program gate, program runner, memory |
| L1 Perception and navigation | Linux, NPU and CPU | 10–30 Hz | Detector, object map, SLAM, costmap, path planning, velocity control |
| L0 Reflex | STM32H5 microcontroller | 1 kHz | Wheel speed control, odometry, safety governor, watchdog |

### One instruction, end to end

1. **Hear.** The microphone array picks up the wake word. Speech-to-text turns the utterance into text.
2. **Gather context.** The agent builds a prompt: the API card, a table of known objects with distance and bearing, robot state, and up to three similar past programs.
3. **Write.** The planner model returns a short spoken acknowledgement and a Python program.
4. **Gate.** The program is parsed, checked against a whitelist, and dry-run in a 2D simulator on the current map. A failed check goes back to the planner with the error.
5. **Run.** The runner executes the program in a sandbox. Each API call becomes a navigation goal, a velocity command, a perception query or speech.
6. **Move.** L1 turns goals and paths into a 20 Hz velocity command, steering around obstacles. L0 turns velocity into wheel speeds and enforces limits.
7. **React.** Detections, pose and safety events flow up continuously. A lost target or blocked path raises an exception inside the program.
8. **Repair or report.** An unhandled exception stops the robot and returns the program, the error and a fresh scene to the planner for a patch. On completion the robot speaks the result.
9. **Learn.** The instruction, final program and outcome are stored and retrieved as examples for similar instructions later.

## 4. Hardware

The body is a round, two-wheel differential-drive base about 300 mm across, carrying a 360° lidar, a forward depth camera, two rear cameras, a microphone array and a 12.8 V battery.

Differential drive on a round footprint is chosen because it turns in place, cannot snag when rotating, and is the best-supported base in ROS 2 navigation.

### The brain board (fixed)

| Part of the [Arduino Ventuno Q](https://www.arduino.cc/product-ventuno-q) | Specification | Used for |
| --- | --- | --- |
| Processor | Qualcomm Dragonwing IQ-8275, 8-core Kryo, Adreno 623 GPU | Linux, ROS 2, SLAM, navigation |
| NPU | Hexagon, up to 40 dense TOPS | Planner model, detector, speech-to-text |
| Memory | 16 GB LPDDR5, shared by CPU, GPU and NPU | All models resident at once |
| Storage | 64 GB eMMC, M.2 slot for NVMe | Models and logs go on an added NVMe drive |
| Microcontroller | STM32H5F5, Cortex-M33 at 250 MHz, 4 MB flash, 1.5 MB RAM, Arduino core on Zephyr | L0 reflex layer |
| Link between the two | Built-in RPC bridge | Velocity commands down, odometry and safety state up |
| Cameras | 3 × MIPI-CSI (4-lane), USB cameras supported | Two rear cameras for 360° coverage |
| USB | 2 × USB 3.0 Type-A, 1 × USB-C | Depth camera, lidar, microphone array through a powered hub |
| Buses | CAN-FD with transceiver on screw terminal, Qwiic I2C, UNO headers, 40-pin header | Qwiic for IMU and power monitor; headers for motors and encoders |
| Network | Wi-Fi 6, Bluetooth 5.3, 2.5 Gb Ethernet | Debug UI and development only |
| Power in | 7–24 V on screw terminal, 12–24 V barrel jack, or 5 V 3 A USB-C | Fed straight from the battery on the screw terminal |
| Size | 160 × 100 × 25.8 mm | Sets the minimum deck size |

Board figures are from the Arduino product page, [Arduino docs](https://docs.arduino.cc/hardware/ventuno-q/) and the [CNX Software review](https://www.cnx-software.com/2026/08/25/299-arduino-ventuno-q-sbc-combines-qualcomm-dragonwing-iq8-soc-and-stm32h5-mcu/).

### Parts chosen for the rest of the robot

Part numbers below are defaults picked from general knowledge, except where linked. Any part meeting the stated requirement is acceptable.

| Function | Default part | Requirement it must meet | Connects to |
| --- | --- | --- | --- |
| Drive motors | 2 × 12 V brushed gearmotor with quadrature encoder, about 150 rpm at the wheel (for example Pololu 37D 70:1) | Top speed ≥ 0.6 m/s on 100 mm wheels; encoder ≥ 1,000 counts per wheel turn | Motor driver; encoder lines to MCU |
| Motor driver | Dual H-bridge, PWM + direction inputs (for example Cytron MDD10A) | 3.3 V logic, ≥ 5 A per channel continuous, 20 kHz PWM | MCU PWM and GPIO pins |
| Wheels and casters | 2 × 100 mm rubber wheels on the centre axis, 2 × ball casters front and rear | Track width about 240 mm | Chassis |
| Lidar | [Slamtec RPLIDAR C1](https://www.slamtec.com/en/c1/spec): 360°, 12 m range, 10 Hz, 5 kHz sampling, 110 g | Unobstructed 360° view on the top deck | USB-UART adapter to Linux |
| Depth camera | Intel RealSense D435i, or Orbbec Gemini 335 | Aligned colour and depth at 640 × 480, 30 fps, usable from 0.3 m to 5 m, ROS 2 driver for arm64 | USB 3.0 to Linux |
| Rear cameras | 2 × wide-angle MIPI-CSI (IMX577 or IMX219), facing ±120° | 720p at 15 fps; with the front camera, no gap around the robot | CSI |
| IMU | BNO085 breakout | Fused orientation at ≥ 100 Hz | Qwiic I2C to MCU |
| Microphones and speaker | [ReSpeaker XVF3800 USB 4-mic array](https://www.seeedstudio.com/ReSpeaker-XVF3800-USB-Mic-Array-p-6488.html) with a 5 W speaker on its amplifier output | Echo cancellation, beamforming, direction of arrival, 5 m pickup | USB to Linux |
| Bumper | Front bumper ring on 2 microswitches | Triggers before the chassis touches | MCU interrupt pins |
| Cliff sensors | 2 × VL53L1X time-of-flight, pointing down at the front | Only needed if the space has stairs or ledges | Qwiic I2C to MCU |
| Emergency stop | Latching mushroom button | Cuts motor driver power in hardware; a second contact tells the MCU | In series with motor supply |
| Battery | 4S LiFePO4 pack, 12.8 V, 10 Ah, with BMS | 10 A continuous; LiFePO4 for indoor safety and a flat voltage that suits 12 V motors | Fuse, main switch, then board and motor driver |
| Power monitor | INA228 breakout | Voltage and current over I2C | Qwiic I2C to MCU |
| USB hub | Powered USB 3.0 hub, 4 ports, fed from a 5 V 5 A buck converter | Camera, lidar adapter and mic array together exceed two ports | USB-A on the board |
| Storage | NVMe SSD, 256 GB or more | Check the slot's M.2 key and length first; not stated on the pages read | M.2 slot |

Approximate cost of everything except the board is USD 600–800, from memory of typical prices, not from quotes.

### Mechanical layout

- **Bottom deck:** motors, driver, battery low and centred, casters, bumper, cliff sensors.
- **Middle deck:** Ventuno Q with its cooler in free air, USB hub, buck converter, emergency stop on the rear edge.
- **Top deck:** lidar at the centre and highest, depth camera at the front edge 250–300 mm above the floor tilted 10° down, microphone array flat on top away from the motors, rear cameras on the mast just below the lidar facing ±120°.

### Power

The battery feeds the board directly through the 7–24 V screw terminal, and the motor driver through the emergency stop. The stop button never cuts power to the computer, so the robot can still report what happened.

Runtime target is about 3 hours, assuming a 40 W average draw. Board power draw is not published on the pages read, so measure it in phase 0.

### Verify in phase 0 before wiring

These were not stated on the pages read. Take them from the [pinout PDF](https://docs.arduino.cc/resources/pinouts/ABX00181-full-pinout.pdf) and the board itself.

- [ ] GPIO logic level of the MCU headers (assumed 3.3 V).
- [ ] Which pins support timer encoder mode. If none are exposed, count encoder edges by interrupt.
- [ ] Name, call syntax and achievable message rate of the RPC bridge.
- [ ] Shipped Ubuntu version, which decides the ROS 2 distribution.
- [ ] Board power draw idle and under full AI load, and whether it throttles.

## 5. Compute budget

Everything fits in about 9 GB of the 16 GB, with one 2B-class vision-language model, one detector and speech models all resident. The NPU is the contended resource, not memory.

| Workload | Default model or package | Runs on | Target rate or latency | Target RAM |
| --- | --- | --- | --- | --- |
| Wheel control, odometry, safety | Custom firmware | MCU | 1 kHz | n/a |
| SLAM and localisation | slam\_toolbox | CPU | 10 Hz | 0.5 GB |
| Costmaps, path planning, control | [Nav2](https://docs.nav2.org/) with MPPI or Regulated Pure Pursuit | CPU | 20 Hz | 0.5 GB |
| Open-vocabulary detection | [YOLOE](https://arxiv.org/abs/2503.07465) small, with masks | NPU | ≥ 10 Hz front, 3 Hz per rear camera, at 640 px | 0.3 GB |
| Tracking and 3D lifting | Box tracker plus depth lookup | CPU | Every frame | 0.1 GB |
| Image and text embeddings | [CLIP ViT-B/16](https://aihub.qualcomm.com/models/openai_clip) or a MobileCLIP variant | NPU | Once per new object | 0.5 GB |
| Planner and visual questions | [LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B), QAD Q4_0 | NPU | ≥ 20 tokens/s; plan ≤ 6 s; yes/no visual answer ≤ 2 s | 2 GB |
| Speech-to-text | [Whisper small](https://aihub.qualcomm.com/models/whisper_small) | NPU | ≤ 1.5 s per utterance | 0.6 GB |
| Wake word and voice activity | openWakeWord, Silero VAD | CPU | Continuous | 0.1 GB |
| Text-to-speech | Piper | CPU | Faster than real time | 0.1 GB |
| Ubuntu, ROS 2, camera and map buffers |  | CPU |  | 3–4 GB |

All rates and sizes above are targets to confirm in phases 0 and 3.

### What is already known about this NPU

- Arduino lists Qwen 3 4B (text and vision), Qwen 2.5 7B vision, Gemma 4 E2B and E4B, Whisper, Piper, Melo and YOLOX as supported, with llama.cpp and Qualcomm's GenieX as runtimes.
- Qualcomm AI Hub lists Qwen3-VL-4B-Instruct, [YOLO-World](https://aihub.qualcomm.com/models/yolo_world), Whisper small, CLIP and Depth Anything V2 for the IQ-8275. No latency is published for any of them.
- The one published measurement is [Qualcomm's guide for Gemma 4 E2B](https://dragonwingdocs.qualcomm.com/blogs/gemma-litert-lm-on-iq8) on this chip under Ubuntu 24.04: about 25–28 tokens/s decode, about 1,240 tokens/s prompt processing, 0.08 s to first token.
- That figure needs a two-line source patch to enable burst mode on Linux. Without it, decode is about 16 tokens/s.
- That build has a 4,096-token context compiled in, covering prompt plus output. It was demonstrated text-only; image input on the NPU was out of scope.
- **Amended, measured by a third party on a VENTUNO Q** ([vq-fast-lfm](https://github.com/heath0xFF/vq-fast-lfm)): the board exposes one usable NPU unit (`HTP1` is rejected), so every NPU model shares it. Writing speed is capped by memory bandwidth, about 45 GB/s divided by the model's size. LFM2.5-2.6B wrote 22.5 tokens/s and read about 690 tokens/s through llama.cpp's Hexagon backend, Q4_0 was the only format at full speed, and a vision encoder ran on the NPU.
- **Amended, measured in this project (M0, laptop GPU, 50 COCO photos):** LFM3-VL-2B answered yes/no questions about real photos correctly 98.3% of the time, picked the right numbered object 80.7% of the time when it checked presence first, and wrote correct programs in a small test language 58.8% of the time with no training.

### Consequences the builder MUST design for

1. **Assume a 4,096-token context.** Budget: 900 for rules and the API card, 700 for the scene table, 700 for retrieved examples, 300 for dialogue, 600 for output. Leave the rest for one low-resolution image when needed.
2. **Plan from text, look only when needed.** The planning call uses the scene table, not images. Images go to the model only through the `ask` primitive, one at a time.
3. **One process owns the planner model.** It serves a priority queue: visual questions from a running program first, planning second, background map naming last.
4. **Arbitrate the NPU.** The detector has priority. If it drops below 8 Hz while the planner decodes, cap the detector at 5 Hz during planning, when the robot is stationary anyway. Lidar safety does not use the NPU.
5. **Keep a CPU fallback.** If a vision-capable NPU build of the planner cannot be made to work, run the same model through llama.cpp on CPU and GPU. Expect several times slower answers.

### Choosing the planner model

**Amended.** The owner chose LFM3-VL-2B. It is the only candidate measured on real photos for this project. At about half the size of a 4B model it writes about twice as fast on a bandwidth-bound NPU, and Liquid built a robot-arm orchestrator on it. Qualcomm's AI Hub page for Qwen3-VL-4B lists this chip but marks the model "not supported" and gives no speed.

**Amended, from the [model card](https://huggingface.co/LiquidAI/LFM3-VL-2B) (checked 2026-11-10).** LFM3-VL-2B reasons step by step before answering ("thinking"). Liquid's own averages drop from 71.4 to 62.1 across 62 benchmarks with thinking off, and agentic tasks from 42.5 to 32.9. Phase 5's first runs had thinking off, so measure it on before fine-tuning, and count the extra tokens against the board's plan-time target. The card also says the model is "not recommended for coding", which is why fine-tuning is expected. Liquid ships SFT and GRPO notebooks for it. It scores 77.3 on visual grounding and 64.4 on embodied reasoning, above Gemma 4 12B on both. Licence: LFM Open License v1.0.

**Amended, measured (2026-11-10, laptop GPU, Q4_0 QAD, 50-case suite).** Thinking off: 19/50 (38%), first plan in 2.0 s and 90 tokens. Thinking on (up to 3,000 characters of reasoning): 9/50 (18%), 24.6 s and 412 tokens. Thinking stays off. Switching the API to degrees and adding `move(metres)` left the score where it was (40% before): the model still writes `drive` for distances and a negative angle for "left". It does not follow the API card's units and signs from the card alone, which is what fine-tuning on examples is for.

**Amended, stack review (2026-11-10). Before fine-tuning, a planner bake-off on the 50-case suite.** Fine-tuning effort should go into the best base model, and three findings make the choice open again:
1. Size matters most for planning. On [PLAN-Bench](https://arxiv.org/html/2610.18836v2) (28 VLMs, embodied planning), Qwen3.5 scores 20.2% at 2B, 43.2% at 4B and 49.2% at 9B. 4B is the knee. LFM3 was not evaluated there.
2. A 4B model fits the board. At Q4_0 it is about 2.5 GB, so roughly 18 tokens/s on a bandwidth-bound NPU, and a 90-token plan takes about 5 s, inside the 8 s target. llama.cpp's Hexagon backend can host several models on the one NPU as virtual sessions of about 3.5 GB each.
3. One LFM3 model covers ears, eyes and voice. [LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B) takes text, images, video and 16 kHz speech, answers in text and/or speech, and is on the same backbone as LFM3-VL-2B: one model in place of speech-to-text, the vision model and text-to-speech. Its card reports 6.36% English WER and 70.6 EmbSpatial; it is non-reasoning (fine here, since thinking measured worse).

Candidates, all measured the same way: LFM3-VL-2B (38–40% now), LFM3-Omni-2B, [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (vision-language, Apache-2.0) and Gemma 4 E4B (Google ships an official Q4_0 QAT GGUF, and Qualcomm has a guide for the E2B on this chip). Each is also tried with the planner calling one primitive per turn as a tool (LFM3 is post-trained for tool calls and its card advises against code), against writing a whole program. Fine-tune the winner, first by SFT on simulator-verified programs and then by GRPO with the suite's checks as the reward: the simulator gives exactly the verifiable reward GRPO needs, and Liquid ships GRPO notebooks.

If LFM3-VL-2B misses the phase 5 target, fine-tune it on simulator-generated programs first. Qwen3-VL-4B and Gemma 4 E4B stay as fallbacks behind the same `Planner` interface.

## 6. Perception and spatial memory

The robot knows its surroundings through three things kept continuously up to date: a 2D occupancy map from lidar, a list of named objects with 3D positions, and a record of which floor area the camera has actually seen.

### Frames and units

- `map` is fixed to the world. `odom` drifts smoothly. `base` is the robot centre with x forward and y left.
- Metres, seconds and radians everywhere in code. Bearings are in `base`, positive to the left, so ±π is directly behind.

### Geometry

- **Odometry:** the MCU integrates wheel encoders and fuses IMU yaw at 100 Hz.
- **Mapping:** slam\_toolbox builds the occupancy grid from lidar and odometry and publishes `map → odom`. The map is saved and reloaded between runs.
- **Obstacles:** the Nav2 costmap combines lidar with depth-camera points, so low objects under the lidar plane are still avoided.

### Open-vocabulary objects

The detector runs on every camera frame with a vocabulary that changes at run time.

- **Live vocabulary.** About 100 common indoor nouns are always active. The `watch` primitive adds task phrases. Text embeddings are computed on the CPU and passed to the NPU graph as an input, so a new phrase takes effect in under 1 s with no recompile.
- **Detector choice.** YOLOE is the default: one model for text-prompted detection and masks, with text prompts folded into the head at no inference cost. YOLO-World is the fallback, already listed on AI Hub for this board under GPL-3.0. Confirm the YOLOE code licence before distributing anything.
- **Amended, detector research (2026-11-10).** The current YOLOE is [YOLOE-26](https://docs.ultralytics.com/models/yoloe/) (zero-shot LVIS minival: 26s 30.8, 26m 35.4, 26l 37.8 AP; AGPL-3.0). Its stock export bakes the classes into the file, and changing them means exporting again, so the "embeddings as a graph input" above needs a custom export of the region-text head. [WeDetect](https://github.com/WeChatCV/WeDetect) (CVPR 2026) is the challenger. Its "dual" ONNX export already takes text features as an input of the vision graph, which is exactly this design. Its proposal variant (WeDetect-Uni) stores class-specific region embeddings, so a later "where are my keys?" can search objects seen before anyone named them. Against it: class names are Chinese for the base model (the planner model can translate), the repository has no licence file, and there is no NPU export yet. Phase 3 measures both on the M0 photos and on the board, and keeps the winner. Not chosen: DINO-X (weights not released as far as found; cloud API), SAM 3.1 (data-centre GPU class, research licence), Mirage (22 MB, objects with attributes and relations; four days old with only vendor numbers; may be added later to fill attributes if it beats `ask`).
- **Propose fast, verify slowly.** Detectors are unreliable on attributes such as colour or state. For "yellow box" the detector proposes every box at 10 Hz, and the planner model confirms "yellow" on the crop once per object.
- **3D lifting.** The median depth inside each mask gives a point in the camera frame, transformed to `map`. Mask extent gives a radius.
- **Rear cameras (amended).** They have no depth. A rear detection takes its range from the lidar return in that direction, or from a floor-plane estimate marked low confidence, and is confirmed when the front camera sees it.

### Object map

One record per physical object, stored in SQLite with embeddings in a NumPy file.

| Field | Meaning |
| --- | --- |
| `id` | Stable integer |
| `label`, `embedding` | Best detector phrase and the image embedding of the best crop |
| `position`, `radius` | Centre in `map` and approximate footprint radius |
| `last_seen`, `n_obs`, `confidence` | Recency, observation count, and a score that decays when the object should be visible and is not |
| `crop_path` | Best image, used by `ask` |
| `facts` | Cached verified answers, for example `{"yellow": true}` |
| `room` | Region id from the map segmentation |

A new detection merges into an existing record when labels are similar and centres are closer than the larger of 0.5 m or the radius. Otherwise it creates a record. Moved objects are handled by the confidence decay, then re-created where next seen.

This is deliberately lighter than dense open-vocabulary scene graphs, which in published systems need a desktop GPU.

### The scene table

The planner never sees raw coordinates alone. The agent renders nearby and task-relevant objects as text with the egocentric direction already worked out:

```
id  label        dist   bearing  side          seen
12  box          1.8 m  -165°    behind-right  4 s ago
7   chair        2.4 m   +30°    front-left    now
3   person(user) 1.1 m    +5°    ahead         now
```

This is what makes "behind you" reliable with a small model: the words come from code, not from the model's arithmetic.

### Rooms and coverage

- **Rooms.** The occupancy grid is split into regions at openings narrower than 1.1 m. Each region gets an id, and a name from the planner model when idle.
- **Camera coverage.** A second grid marks floor cells the camera has viewed within 4 m. Lidar sees all round but the camera does not, so "searched" means camera-covered.
- **Unseen viewpoints.** The `unseen` primitive returns reachable poses that would reveal the most uncovered area in a region, nearest first.

### People

The `person` class is always in the vocabulary and tracked. The speaker is the person track closest to the microphone array's direction of arrival when the wake word fires. That position is stored as the user.

## 7. Cognition: planning without predefined skills

For every instruction the planner model writes a new Python program against the primitive API, and that program is the behaviour. Nothing in the codebase knows what a "circle" or a "search" is.

### The agent loop

| Step | Component | What it does | Budget |
| --- | --- | --- | --- |
| 1 | Context builder | Assembles rules, API card, scene table, robot state, up to 3 retrieved programs, dialogue, the instruction | 0.1 s |
| 2 | Planner | Returns JSON: `say_first`, `plan` (one sentence), `program` | ≤ 8 s |
| 3 | Gate | Static checks, then a dry run in the 2D simulator | ≤ 0.5 s |
| 4 | Runner | Executes the program in a sandboxed process | Task length |
| 5 | Monitor | Turns safety events, lost targets, blocked paths and timeouts into exceptions | 20 Hz |
| 6 | Repair | On an unhandled exception: stop, re-prompt with program, traceback and new scene table | ≤ 3 per task |
| 7 | Report and store | Speaks the outcome; saves instruction, program and result | 0.1 s |

### Planner output

The planner MUST return exactly this shape. Use grammar-constrained decoding if the runtime offers it; otherwise parse and retry once.

```json
{
  "say_first": "Okay, three laps around the box.",
  "plan": "Find the box behind me, build a 3-lap circular path around it, follow it.",
  "program": "...python source..."
}
```

### The gate

No program moves the robot until it passes all three checks. A failure returns the error text to the planner, at most twice. After that the robot says it cannot do the task and why.

1. **Syntax and whitelist.** Parse to an AST. Allow only `robot`, `geom`, `math`, basic Python control flow, comprehensions, local functions, and a fixed list of safe builtins (len, range, min, max, abs, sum, sorted, round, any, all, enumerate, zip, set, list, dict, str, int, float, bool). Reject imports, attribute names starting with an underscore, `eval`, `exec`, `open` and anything not on the list.
2. **Preemptibility.** Every loop body MUST contain at least one `robot` call, so the runner can interrupt it.
3. **Dry run.** Execute against a simulator backend that implements the same API on the current occupancy map and object map. Reject programs that raise, exceed 2,000 API calls, or command a path through occupied cells. **Amended:** the dry run stops after 2,000 calls or 120 simulated seconds, and a program still running cleanly at that point passes. Rejecting it would reject every "follow me until I say stop" and every 20 Hz control loop this section recommends.

### The runner

- Runs the program in a separate process as an unprivileged user with no network or file access, talking to the API over a local socket.
- Every API call is a checkpoint. At each one the runner can raise `Preempted`, `SafetyStop`, `PathBlocked`, `TargetLost`, `NotFound` or `Timeout` inside the program.
- A spoken "stop" kills the process and zeroes velocity without consulting the planner.
- Wall-clock limit per task defaults to 10 minutes.

### Where real-time behaviour comes from

A program MAY loop at up to 20 Hz, reading `robot.detect` or `robot.pose` and calling `robot.drive`. That loop is the closed-loop controller for the task. Following a moving person, keeping a fixed distance from a wall, or stopping the moment something appears all run this way, with zero model calls per cycle.

### Repair and verification

- **Repair.** The repair prompt contains the original instruction, the program, the line that failed, the exception and the new scene table. The planner returns a full replacement program that continues from the current state.
- **Verification.** When success is visual, the program SHOULD confirm with `robot.ask` on a fresh frame before reporting success. The planner rules say so.
- **Clarification.** If the scene table shows two equally good referents, the program asks the user with `robot.say` and `robot.listen` instead of guessing.

### Program memory

Skills are learned, not shipped. Every completed task stores its instruction, final program, outcome and any spoken feedback. The context builder retrieves the three most similar successful programs by embedding similarity and includes them as examples.

The robot starts with about six seed examples that teach API style only: one perception query, one motion call, one loop with an exception handler, one question to the user. No seed may solve a task in the acceptance suite.

### Planner interface

```python
class Planner(Protocol):
    def plan(self, context: Context) -> PlanResult: ...
    def repair(self, context: Context, failure: Failure) -> PlanResult: ...
    def answer(self, question: str, image: Image | None) -> str: ...
```

`LocalPlanner` wraps the on-board model and is the only implementation enabled by default. A `RemotePlanner` for a cloud reasoning model MAY be added behind the same interface and MUST be off unless the owner enables it. Its only intended use is escalation after the local planner fails the gate twice.

## 8. Robot API

This is the complete surface the planner writes against: 24 calls on `robot` and 4 pure-math helpers on `geom`. The same interface has two backends, the real robot and the simulator.

```python
# Injected names: robot, geom, math. No imports.
# Frames: positions are in map. Bearings are in base, + is left, ±pi is behind.
# Units: metres, seconds, radians.

class Robot:
    # --- state -----------------------------------------------------------
    def pose(self) -> Pose                        # x, y, yaw in map
    def battery(self) -> float                    # 0.0 .. 1.0
    def user(self) -> Object | None               # last person who spoke
    def room(self) -> str                         # id of the region the robot is in

    # --- motion: blocking, preemptible, always safety-filtered -----------
    def drive(self, v: float, w: float, t: float) -> None   # forward m/s, turn rad/s, for t seconds
    def turn(self, angle: float) -> None                    # relative, + is left
    def face(self, target: Point | Object) -> None
    def go_to(self, target: Point | Pose | Object, stop_dist: float = 0.0) -> None   # planned, avoids obstacles
    def follow(self, path: list[Point], speed: float = 0.3) -> None                  # tracks the path, avoids obstacles
    def stop(self) -> None

    # --- perception ------------------------------------------------------
    def watch(self, *phrases: str) -> None        # add phrases to the live detector vocabulary
    def detect(self, phrase: str) -> list[Object] # in view right now, nearest first
    def find(self, phrase: str) -> list[Object]   # from the object map, best match first
    def look(self) -> Image                       # current camera frame
    def ask(self, question: str, about: Image | Object | None = None) -> str   # model answer, about 2 s
    def scan(self) -> list[tuple[float, float]]   # lidar as (bearing, range)

    # --- map -------------------------------------------------------------
    def is_free(self, p: Point) -> bool
    def unseen(self, region: str = "room") -> list[Pose]   # viewpoints revealing unsearched area, nearest first

    # --- interaction and memory -----------------------------------------
    def say(self, text: str) -> None
    def listen(self, timeout: float = 8.0) -> str | None
    def remember(self, note: str) -> None
    def recall(self, query: str) -> list[str]
    def sleep(self, t: float) -> None
    def elapsed(self) -> float                    # seconds since the program started

class Object:
    id: int
    label: str
    position: Point      # map
    distance: float      # from robot, now
    bearing: float       # base frame, now
    radius: float
    last_seen: float     # seconds ago

class geom:              # pure math, no robot access
    def dist(a, b) -> float
    def bearing(pose, p) -> float
    def circle(center, radius, start, laps=1.0, ccw=True, step=0.15) -> list[Point]
    def line(a, b, step=0.15) -> list[Point]
```

### Semantics the builder MUST implement

- **`go_to` and `follow`** hand a goal or path to Nav2. They return on arrival, and raise `PathBlocked` after 10 s without progress.
- **Paths that cross themselves (amended).** `follow` sends a multi-lap path to Nav2 one lap at a time. Path trackers look for the nearest point on the path, so a three-lap circle sent whole can jump to a later lap and finish early.
- **`drive`** bypasses the path planner but not the collision monitor or the MCU governor. It is for short reactive motions.
- **`detect`** returns only what the camera sees now. **`find`** searches memory by label, then by embedding similarity, so it also answers loose phrases such as "something to sit on".
- **`ask`** with an `Object` uses that object's stored crop. With nothing, it uses the current frame. Answers to yes/no questions are cached in the object's `facts`.
- **Picking among candidates (amended).** When the model has to choose one object from an image, ask first whether the thing is there. In M0, offered "none" as an answer, it still picked a candidate 49 times out of 50 for things that were absent.
- **`watch`** persists for the rest of the program. Detections accumulate in the object map while the robot moves, with no further calls.
- **Exceptions** are part of the API: `PathBlocked`, `TargetLost`, `NotFound`, `Timeout`, `SafetyStop`, `Preempted`. Programs may catch the first four. The last two always end the program.

### Rule for extending the API

A new call is allowed only if it exposes a sensor, an actuator, memory or pure math. A call that encodes a task is forbidden. `geom.circle` returns points; it does not move the robot.

The API card given to the planner is this listing with comments, kept under 700 tokens. Generate it from the code so the two cannot drift.

## 9. Real-time control and safety

The microcontroller is the only thing wired to the motors, and it enforces speed limits, a watchdog and bumper stops no matter what Linux or the AI sends.

### Firmware on the STM32H5 (L0)

| Task | Rate | Behaviour |
| --- | --- | --- |
| Wheel speed control | 1 kHz | Read encoders, PI plus feed-forward per wheel, write PWM |
| Odometry | 100 Hz | Integrate wheel motion, fuse IMU yaw, send pose, velocity and safety state to Linux |
| Command intake | On arrival | Accept `(v, w)`, clamp to limits, apply acceleration ramp |
| Watchdog | 1 kHz check | No valid command for 250 ms ramps both wheels to zero |
| Bumper | Interrupt | Stop within one control cycle; refuse forward motion until released |
| Cliff | 50 Hz | Stop and refuse forward motion while a drop is seen |
| Battery | 10 Hz | Below 11.0 V refuse motion and report; values to tune for the pack |
| Status | 10 Hz | Drive the on-board LED matrix: idle, listening, thinking, moving, fault |

Firmware limits are compile-time constants: 0.5 m/s forward, 1.5 rad/s turn, 0.8 m/s² acceleration. These are starting targets. Linux can request lower limits at run time, never higher.

### Bridge between Linux and the MCU

- Four calls are enough: `set_velocity(v, w)`, `get_state()`, `set_limits(v_max, w_max)`, `set_status(code)`.
- Linux sends velocity at 20 Hz or faster, which also serves as the heartbeat.
- The built-in RPC bridge is the default transport. Its rate is not documented on the pages read. If it cannot carry 50 Hz state with under 10 ms round trip, replace it with a framed binary protocol on a UART between the two processors.

### Safety on Linux (L1)

- **Collision monitor.** Nav2's collision monitor sits between every velocity source and the bridge. It slows inside 0.6 m and stops inside 0.25 m of any lidar or depth obstacle in the direction of travel.
- **Single velocity path.** `drive`, `go_to` and `follow` all publish to one topic that passes through the velocity smoother and collision monitor. There is no second route to the bridge.
- **People.** Maximum speed halves when a tracked person is within 1.5 m.

### Safety rules

- **S1.** Every motion command from a generated program MUST pass through the collision monitor and the MCU governor.
- **S2.** Loss of the Linux heartbeat MUST stop the wheels within 250 ms.
- **S3.** The emergency stop MUST cut motor power in hardware, with no software in the path.
- **S4.** Speed and acceleration ceilings MUST live in firmware and MUST NOT be raisable over the bridge.
- **S5.** Generated programs MUST NOT have file, network or process access.
- **S6.** The spoken word "stop" MUST be handled by a keyword detector that preempts the runner directly.
- **S7.** The robot MUST NOT move for a program that has not passed the gate.

### Reaction-time targets

| Event | Handled by | Target |
| --- | --- | --- |
| Bumper pressed | MCU | ≤ 5 ms to zero PWM |
| Obstacle enters stop zone | Collision monitor | ≤ 150 ms |
| Linux process dies | MCU watchdog | ≤ 250 ms |
| Spoken "stop" | Keyword detector to runner | ≤ 500 ms |
| Program exception to robot stationary | Runner | ≤ 100 ms |

## 10. Speech and interaction

The robot is addressed by voice through a wake word, answers by voice, and can also be driven and inspected from a local web page.

1. **Capture.** The microphone array does echo cancellation, beamforming and noise suppression in its own hardware, and reports direction of arrival.
2. **Wake.** openWakeWord listens continuously on the CPU. On wake, the robot plays a short tone and turns to face the direction of arrival.
3. **Endpoint.** Silero VAD closes the utterance after 0.7 s of silence.
4. **Transcribe.** Whisper small on the NPU returns text and language.
5. **Speak.** Piper synthesises replies on the CPU. Echo cancellation lets the robot hear "stop" while it is talking.

- **Fast feedback.** The tone and the turn happen within 0.5 s of the wake word. The planner's `say_first` line is spoken as soon as planning ends, before the gate finishes.
- **Stop word.** A second always-on keyword model listens for "stop" with no wake word needed, as rule S6 requires.
- **Amended.** Phase 6 also measures Moonshine (speech-to-text built for small devices) against Whisper, and Kokoro against Piper by listening on the board.
- **Language.** Whisper and Piper are multilingual. The spoken language is a config value, and the planner prompt and API card stay in English.
- **Web page.** Served on the robot over Wi-Fi: typed instructions, live camera with detections, map with objects, the current program with its executing line, and the task log. This is the main debugging tool; build it in phase 4.

## 11. Worked examples

Both tasks from the brief reduce to programs of under 25 lines over the API in section 8. These are the kind of program the planner is expected to write; they are not shipped with the robot.

### "Do 3 circles around the box that's behind you"

```python
robot.watch("box")
behind = [o for o in robot.find("box") if abs(o.bearing) > math.radians(110)]
if not behind:                        # nothing remembered behind me: turn and look
    robot.turn(math.pi)
    behind = robot.detect("box")
if not behind:
    robot.say("I can't see a box behind me.")
else:
    box = min(behind, key=lambda o: o.distance)
    r = box.radius + 0.45             # robot radius 0.15 + 0.30 clearance
    path = geom.circle(box.position, r, start=robot.pose(), laps=3)
    robot.follow(path, speed=0.3)
    robot.say("Done. Three circles around the box.")
```

| Time | What happens | Layer |
| --- | --- | --- |
| 0.0 s | Utterance ends. Robot is already facing the speaker | L2 |
| 1.5 s | Text ready. Scene table shows `box 1.8 m behind-right` | L2 |
| 8 s | Program written, about 150 tokens. Robot says "Okay, three laps around the box." | L2 |
| 8.5 s | Gate passes: dry run traces three clear laps on the map | L2 |
| 9 s | `follow` starts. Controller tracks the path at 20 Hz; wheels at 1 kHz | L1, L0 |
| 9–55 s | Three laps of about 4.4 m each at 0.3 m/s. No model calls | L1, L0 |
| any | Someone steps into the path: collision monitor stops, controller steers round, or `PathBlocked` after 10 s triggers a repair | L1, L2 |
| 56 s | Robot says "Done." Program and outcome stored | L2 |

Times are targets built from the budgets in section 5.

### "Search for a yellow box in this room. If you find it, come back to me and tell me"

```python
home = robot.user() or robot.pose()
robot.watch("box")
checked, found = set(), None
for _ in range(25):                               # bounded search
    for o in robot.find("box"):
        if o.id in checked:
            continue
        checked.add(o.id)
        if robot.ask("Is this box yellow? Answer yes or no.", o).lower().startswith("yes"):
            found = o
            break
    views = robot.unseen("room")
    if found or not views:
        break
    robot.go_to(views[0])                         # detector keeps filling the map on the way
if found:
    where = robot.ask("In a few words, where is this box?", found)
    robot.go_to(home, stop_dist=1.0)
    robot.face(home)
    robot.say("I found a yellow box. " + where)
else:
    robot.say("I searched the room and did not find a yellow box.")
```

What each requirement in the sentence maps to:

- **"search"** is a loop over `unseen` viewpoints. Coverage is tracked by L1, so the loop ends when the camera has seen the whole room.
- **"yellow box"** is split: the detector proposes every box at 10 Hz while driving, and one `ask` per new box confirms the colour.
- **"this room"** is the region argument to `unseen`, bounded by the room segmentation.
- **"come back to me"** uses the user position stored when the wake word fired.
- **"if you find it"** is ordinary Python control flow, which is why the plan is code and not a fixed list of steps.

If a doorway is blocked mid-search, `go_to` raises `PathBlocked`. The repair call sees the program, the failed line and the new scene, and typically returns a program that skips that viewpoint and continues.

## 12. Software stack

The Linux side is ROS 2 nodes in Python and C++ on the board's Ubuntu, and the MCU side is one Arduino sketch on Zephyr.

- **OS.** Ubuntu as shipped. Qualcomm's NPU measurement on this chip used Ubuntu 24.04. **Amended:** the development laptop's WSL runs Ubuntu 26.04, so ROS 2 Jazzy runs there in a 24.04 distribution or in Docker.
- **ROS 2.** Jazzy if the board runs Ubuntu 24.04. Nav2 lists Jazzy and Lyrical as actively supported. If the host version does not match, run ROS 2 in Docker, which the board supports.
- **AI runtimes.** Qualcomm's runtime for NPU models (AI Hub exports, GenieX or LiteRT-LM, whichever carries the chosen model), llama.cpp as CPU and GPU fallback.
- **Arduino App Lab.** Use it to flash the sketch and to check the stock AI examples in phase 0. The robot itself does not depend on it.

### Processes

| Process | Role | Language | Publishes or serves |
| --- | --- | --- | --- |
| `firmware` | L0 on the MCU | C++ sketch | Bridge calls in section 9 |
| `mcu_bridge` | Bridge to ROS | Python or C++ | `/odom`, `/imu`, `/safety_state`; consumes `/cmd_vel_safe` |
| `lidar`, `camera` | Vendor drivers | C++ | `/scan`, `/color`, `/depth`, `/points` |
| `slam` | slam\_toolbox | C++ | `/map`, `map → odom` |
| `nav` | Nav2 planner, controller, smoother, collision monitor | C++ | Actions `navigate_to_pose`, `follow_path`; `/cmd_vel_safe` |
| `detector` | Open-vocabulary detection and tracking on the NPU | Python | `/detections`; service `set_vocabulary` |
| `object_map` | Object records, rooms, coverage grid | Python | Services `find`, `unseen`, `scene_table` |
| `speech` | Wake word, stop word, VAD, speech-to-text, text-to-speech | Python | `/utterance`, `/stop_word`; service `say` |
| `model_server` | Owns the planner model; priority queue | Python | Services `plan`, `repair`, `answer` |
| `agent` | Context builder, gate, runner supervisor, program memory | Python | Action `do_instruction` |
| `runner` | Sandboxed child process executing one program | Python | Local socket to `agent` only |
| `web_ui` | Debug and typed-command page | Python | HTTP on the local network |

### Repository layout

```
robot/
  firmware/            Arduino sketch: control loop, governor, bridge handlers
  ros_ws/src/
    robot_bringup/     launch files, robot.yaml, URDF, Nav2 and SLAM params
    mcu_bridge/
    detector/
    object_map/
    speech/
    model_server/
    agent/
      api/             Robot and geom interfaces, real backend, sim backend
      gate/            AST whitelist, preemptibility check, dry run
      runner/          sandbox process, exception injection
      memory/          program store and retrieval
      prompts/         rules, seed examples, generated API card
    web_ui/
  sim/                 2D kinematic simulator; optional Gazebo world
  eval/                instruction suite, red-team programs, scoring
  models/              export and quantisation scripts, model manifests
  docs/                this report, phase 0 findings, measured budgets
```

### Rules for the codebase

- **One config file.** `robot.yaml` holds dimensions, limits, model names, rates and thresholds. No magic numbers elsewhere.
- **Simulator first.** Everything in `agent/` MUST run on a development PC against the sim backend with no robot attached. The gate's dry run reuses that backend.
- **Task records.** Every instruction writes one JSON record: instruction, context, each program version, gate results, events, timings, outcome. Program memory and debugging both read these.
- **Model manifests.** Each model has a manifest with source, licence, export command, input shapes and measured latency on the board.

## 13. Build phases and acceptance tests

Build in eight phases. Each ends in tests that MUST pass on the real robot unless marked sim, and each leaves a robot that does something useful.

| Phase | Build | Acceptance tests |
| --- | --- | --- |
| 0. Board bring-up | Flash and update the board. Resolve the checklist in section 4. Run LFM3-VL-2B on the NPU through llama.cpp's Hexagon backend and repeat M0 there. Run a stock detector on the NPU. | Checklist answered in `docs/`. LFM3-VL-2B writes ≥ 20 tokens/s and M0 scores hold on the board. MCU toggles a pin from a Linux call, with round-trip time measured. |
| 1. Base mobility | Chassis, wiring, firmware control loop, governor, bridge, keyboard teleop. | 2 m straight: ≤ 3 cm sideways error, ≤ 2% length error. 360° turn: ≤ 3° error. Killing the bridge process stops wheels in ≤ 250 ms. Bumper and emergency stop both stop the robot. |
| 2. Mapping and navigation | Lidar, depth camera, SLAM, Nav2, collision monitor. | Maps one room and reloads the map. 10 of 10 goals reached within 10 cm. Stops for a box placed in its path, then drives round it. |
| 3. Perception and object map | Detector on NPU with live vocabulary, tracking, 3D lifting, object map, rooms, coverage. | Detector ≥ 10 Hz. Five objects at measured spots mapped within 20 cm. New phrase active in ≤ 1 s. A moved object is found at its new place after one pass. |
| 4. API, simulator, gate, runner | Both API backends, gate, sandbox, web page. Hand-written test programs only. | Ten hand-written programs behave the same in sim and on the robot. Gate rejects 20 of 20 red-team programs. Preempt to stationary ≤ 300 ms. |
| 5. Planner | Model server, context builder, prompts, seed examples, repair loop. LFM3-VL-2B, fine-tuned on simulator programs if it misses the target. | Sim: ≥ 80% of the 50-instruction suite passes with ≤ 2 repairs each. Plan latency ≤ 8 s at the median on the board. |
| 6. Speech and full loop | Wake word, stop word, speech-to-text, text-to-speech, user tracking. | Both tasks from the brief succeed in 4 of 5 spoken trials each, in a room not used during development. Spoken "stop" halts in ≤ 500 ms. |
| 7. Memory and hardening | Program memory, task records, thermal and endurance checks. | A repeated instruction plans faster or passes more often with memory on. One hour of mixed tasks with no crash and no thermal throttling of the control loop. |

### The 50-instruction suite

Write it in phase 4, before any planner work, so it cannot be fitted to the model. Each entry is an instruction, a simulator scene and an automatic success check.

| Category | Count | Example |
| --- | --- | --- |
| Relative motion | 8 | "Back up one metre, then spin twice" |
| Object-relative motion | 10 | "Drive a figure of eight around the two chairs" |
| Search and report | 8 | "Is there a backpack anywhere in here?" |
| Conditional and counting | 8 | "If the door is open go through it, otherwise tell me" |
| Following and reactive | 6 | "Follow me until I say stop" |
| Memory | 5 | "Where did you last see my keys?" |
| Ambiguous or impossible | 5 | "Go to the box" with three boxes in view |

The two tasks from the brief are held out of all prompts and seed examples and are tested only in phase 6.

## 14. Risks, open questions and upgrade paths

The largest risk is that a 4B-class on-board model writes wrong programs too often; the gate, the repair loop, program memory and the model bake-off all exist to contain that.

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Planner writes incorrect code | "Any task" depends on it. **Amended:** untrained, the 2B model wrote correct programs 59% of the time in M0 | Short API, scene table in words, gate with dry run, up to 3 repairs, retrieved examples, fine-tuning on simulator programs, optional remote planner |
| Vision input not available on the NPU build | The only public measurement on this chip is text-only | Use llama.cpp's Hexagon backend, where a vision encoder has already run on this NPU; fall back to CPU and GPU for `ask` |
| 4,096-token context | Limits scene size and examples | Token budget in section 5; scene table capped at 25 objects, ranked by relevance |
| NPU contention | Detector and planner share one NPU | Detector priority, 5 Hz cap during planning, measure in phase 3 |
| Bridge rate unknown | Control loop needs 50 Hz state | Measure in phase 0; UART fallback |
| Encoder pins unknown | Affects odometry quality | Pinout check in phase 0; interrupt counting fallback |
| "This room" is fuzzy | Open-plan spaces have no door-width gaps | Fall back to a 4 m radius around the user and say so |
| Colour and attribute errors | Detector proposals are attribute-blind | Verify with `ask`; cache answers per object |
| Glass, mirrors, black furniture | Lidar and depth both miss them | Bumper as last line; allow no-go zones drawn on the web page |
| Licences | YOLO-World on AI Hub is GPL-3.0; YOLOE code licence unconfirmed | Fine for a private build; check before any distribution |
| Heat | Sustained NPU load may throttle | Cooler in free air; endurance test in phase 7 |

### Decisions for the owner

- [ ] On-board only, or allow a cloud model as an escalation path when the local planner fails twice?
- [x] Rear cameras: two, for 360° coverage (decided 10 November 2026).
- [ ] Is there any stair or ledge in the operating area? If not, drop the cliff sensors.
- [ ] Wake word and spoken language.

### Upgrade paths the architecture already allows

- **An arm.** Add `arm.do(text)` as a primitive backed by a small vision-language-action model. The planner then orchestrates it exactly as it does navigation, the same split Gemini Robotics 1.5 uses.
- **A learned navigator.** A navigation foundation model can replace the inside of `go_to` without changing any program. **Amended, candidate found (2026-11-10):** [LightNav-0](https://huggingface.co/LightOriginsHQ/LightNav-0) (Apache-2.0; Qwen3-VL-4B, 5B parameters) takes 4 fps front-camera video and an instruction, and returns 10 waypoints at a time for a follower to drive. One model covers following directions, finding an object by name and tracking a person ("follow the person in the red shirt"); the authors report the best monocular results on 10 simulated benchmarks and runs on a wheeled robot. It fits as the inside of `follow`-style tracking and of searching for unmapped things, with its waypoints still going through Nav2's controller and collision monitor. Before adopting: measure its step time next to LFM3-VL on the one NPU (it reads several frames each step; about 2.5 GB at 4 bits), and test it on its own Habitat benchmarks. Not before phase 3.
- **A stronger planner.** A larger or better model drops in behind `Planner` as NPU builds appear. Re-run the suite to decide.
- **Amended, why the stack stays hierarchical (2026-11-10).** A real-robot evaluation of vision-and-language navigation ([IEEE TASE 2026](https://arxiv.org/abs/2607.09792)) measured a camera-only end-to-end method at 61% success in simulation and 22% on the real robot, against 51% on the real robot for a hierarchical system. Learned navigators enter as tools of the planner, not as the whole robot. [Qwen-RobotNav](https://huggingface.co/papers/2606.18112) (2B–8B) is built for exactly that, with task modes an outer planner switches (follow directions, find an object, track a person, answer about the scene), but its weights were not found public. [InternVLA-N1](https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN) and LightNav-0 are public.
- **Amended, object memory.** "Where did I leave my keys?" follows the 2026 line of open-vocabulary spatial memory ([DynaMem](https://arxiv.org/html/2411.04999), [STaR](https://arxiv.org/html/2602.09255)): every detection keeps its embedding, a crop and a time, and queries are answered by embedding search, then confirmed by the vision model on the saved crop.
- **A second robot.** Object map and program memory are files; they can be shared.

## 15. Sources

Pages opened on 10 November 2026. Figures in this report marked as targets are estimates, not taken from these pages.

| Topic | Source |
| --- | --- |
| Ventuno Q specifications, supported models and runtimes | [Arduino product page](https://www.arduino.cc/product-ventuno-q), [Arduino docs](https://docs.arduino.cc/hardware/ventuno-q/), [CNX Software review](https://www.cnx-software.com/2026/08/25/299-arduino-ventuno-q-sbc-combines-qualcomm-dragonwing-iq8-soc-and-stm32h5-mcu/) |
| Measured language-model speed on the IQ-8275 NPU | [Qualcomm Dragonwing guide: Gemma on IQ8 with LiteRT-LM](https://dragonwingdocs.qualcomm.com/blogs/gemma-litert-lm-on-iq8) |
| Planner candidates | [Qwen3-VL-4B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) |
| Models listed for this chip | Qualcomm AI Hub: [YOLO-World](https://aihub.qualcomm.com/models/yolo_world), [Whisper small](https://aihub.qualcomm.com/models/whisper_small), [OpenAI CLIP](https://aihub.qualcomm.com/models/openai_clip), [Depth Anything V2](https://aihub.qualcomm.com/models/depth_anything_v2) |
| Open-vocabulary detection | [YOLOE: Real-Time Seeing Anything](https://arxiv.org/abs/2503.07465) |
| Reasoning model orchestrating a lower layer | [Gemini Robotics 1.5 announcement](https://deepmind.google/discover/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/) |
| Programs as robot policies | [Code as Policies](https://arxiv.org/abs/2209.07753), [Agentic Scene Policies](https://arxiv.org/abs/2509.19571) |
| Navigation foundation models | [NavFoM](https://arxiv.org/abs/2509.12129) |
| Navigation stack | [Nav2 documentation](https://docs.nav2.org/) |
| Lidar | [Slamtec RPLIDAR C1 specification](https://www.slamtec.com/en/c1/spec) |
| Microphone array | [Seeed ReSpeaker XVF3800](https://www.seeedstudio.com/ReSpeaker-XVF3800-USB-Mic-Array-p-6488.html) |
| Measured results (amended) | [vq-fast-lfm](https://github.com/heath0xFF/vq-fast-lfm) on a real VENTUNO Q; M0 results in `~/robot-runtime/runs/m0` |

Not opened, and so not relied on for any figure: the Ventuno Q pinout, schematic and power-tree PDFs, the Arduino user manual (returned empty), and the ROS 2 release schedule (blocked). Motor, driver, depth camera, IMU and battery choices come from general knowledge and should be checked against current datasheets.
