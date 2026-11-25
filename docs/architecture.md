# Sprocket v2 Architecture

**Specification — Living Body + 3-CSI Pure Vision + Long-Term Memory + an Agentic Vision-Language Mind**

Sprocket is an indoor, general-purpose wheeled robot on the **Arduino VENTUNO Q**. Its mind is an **agentic tool-calling loop** around an on-board vision-language model, built the way coding agents are: the model sees the cameras, calls tools in its own native format, reads every result, keeps a to-do list for several-part requests, and ends a task by replying in words. Under it sit a body that drives itself between calls, 360° pure-vision senses (no spinning lidar), and long-term memory.

> **Status (19 November 2026).** Sections 4 and 6 (perception and the agent) describe the code as it is. Sections 3 and 5 describe the target design and are only partly built: the wake-word turn reflex and the voice-direction buffer exist but are not wired in; the body runs one skill at a time (a skill's own control loop drives, the model is not asked while it runs); the long-term memory stack is still being decided. What the 3D runs showed is in [simulation_3d.md](simulation_3d.md); how to run everything is in [usage.md](usage.md).

---

## 1. Core Principles

1. **Living, Non-Stop Motion:** The body never freezes into warehouse-cart stop-and-go steps. It rolls out smooth trajectory curves at 30–50 Hz, blends waypoints continuously, yields politely to approaching people, and expresses attention through posture and synthesized chimes.
2. **An Agentic Mind That Actually Sees:** the vision-language model (`LFM3-VL-2B` for now; not final) gets real camera pictures with tracker ids drawn on them, not only a text list of detections. It acts through native tool calls and reads each result before deciding the next step (Section 6).
3. **Yes/no probabilities from the same model, where a skill needs a judgement:** one token's probability answers perception's check of each detection ("is there a tv in the middle of this picture?") and the search's choice of direction ("would going this way lead to a plant?"); the model is never inside a control loop.
4. **Pure 360° Vision + Low-Cost Multi-Zone ToF:** Three wide-angle MIPI-CSI cameras calibrated by floor-plane trigonometry, IMU pitch/roll, and 8×8 `VL53L5CX` laser ToF anchors replace the expensive spinning 2D lidar while catching low obstacles (cables, shoes, pets) below a lidar plane.
5. **Circulating Memory (Nothing Is Set in Stone):** Short-term working memory tracks active goals and recent visual frames in RAM; long-term memory decays by volatility half-life, updates movable habits via EMA, overwrites stale entries instantly when cameras see empty space, and recycles expired records.
6. **The Model Decides, the Harness Informs:** no keyword or regex routing of instructions and no silent rewriting of the model's choices. "Stop" and "wait" are the only words handled without the model.

---

## 2. Hardware & Heterogeneous Compute Budget (Arduino VENTUNO Q)

Brain board: **Arduino VENTUNO Q** (Qualcomm Dragonwing IQ-8275: 8-core Kryo CPU, Adreno 623 GPU, 40 TOPS Hexagon NPU, 16 GB shared LPDDR5 RAM, integrated STM32H5F5 MCU on RPC bridge).

### Sensors & Chassis
- **Base:** 300 mm round two-wheel differential drive (`WHEEL_R = 0.05 m`, `TRACK = 0.24 m`), front/rear ball casters, `BNO085` IMU (100 Hz), `INA228` motor current monitor, hardware e-stop, 4S LiFePO4 battery.
- **360° Pure-Vision Rig (No Lidar):** 3 × wide-angle MIPI-CSI cameras (`1280×720`, `hfov = 2.44 rad / ~140°`) at `CAMERA_Z = 0.22 m`, `CAMERA_R = 0.11 m`:
  - `front_camera`: `0.0 rad` ($0^\circ$)
  - `rear_left_camera`: `+2π/3 rad` ($+120^\circ$)
  - `rear_right_camera`: `-2π/3 rad` ($-120^\circ$)
  - Adjacent cameras overlap by $\approx 20^\circ$ for seamless $360^\circ$ tracking.
- **Multi-Zone ToF Reflex Sensors:** 2 × `VL53L5CX` 8×8 (64-zone) ToF laser arrays (front and rear) on the MCU I2C bus for cliff/step measurement, glass/blank-wall protection, and metric depth laser anchoring.
- **Audio Rig:** 4-microphone array (`ReSpeaker XVF3800`) providing real-time Direction of Arrival (DOA) angle + 5 W speaker for Piper TTS and procedural chimes.

### Compute Allocation (~10 GB / 16 GB RAM)
| Processor | Workloads | Rate / Latency |
| :--- | :--- | :--- |
| **40 TOPS Hexagon NPU** | `LFM3-VL-2B` (Q4_0: the agent, yes/no checks), `YOLOE-26l-seg` (open-vocab masks for the asked words) | not measured on the board yet; on a desktop RX 6600: a picture read in ~0.15 s, YOLOE-26l 75 ms a frame |
| **Adreno 623 GPU** | `Depth Anything V2 small` (relative depth, scaled by the floor), `RTMPose` / `YOLO-Pose` (keypoints/gaze/pointing), `MobileCLIP2` (crop embeddings) | 10–30 Hz total |
| **8-core Kryo CPU** | 30–50 Hz Living Body Engine, `slam_toolbox` / Nav2 costmap, `WeSpeaker` voiceprint embedding, `Whisper`, `Piper` TTS, `ChimeSynthesizer`, SQLite + Vector Memory | 30–50 Hz body, <2 ms memory lookup |
| **STM32H5 MCU** | 200 Hz wheel PID + gyro odometry, `VL53L5CX` 8×8 step/cliff governor, IMU+current anti-cable snag reflex, 300 ms command watchdog | 200 Hz–1 kHz |

---

## 3. Layer 1: The Living Body Engine (`robot/body/controller.py`)

The full trajectory rollout below currently drives the 2D simulator at a 20 Hz simulated timestep.
In ROS, Nav2 owns path following; the API owns direct motion such as gestures. Both use one velocity gate
(`robot/ros/motion.py`): Nav2 and direct motion cannot own it at the same time. The gate sends only the active
owner's command to Collision Monitor and accepts only fresh, matching replies for `/sprocket/cmd_vel_wheels`.
Nav2 docking and following publish on `/cmd_vel`, which the base does not consume. The base, not
Nav2's velocity smoother, now limits acceleration. Stop closes the gate and sends zero before waiting on Nav2.
A late goal response is canceled, and no new motion starts until that goal is terminal. The gate checks for
missing replies at 40 Hz and sends zero once a checked command is more than 250 ms old at its source in ROS time.
Habitat independently enforces the same source-age limit.
A ROS clock rewind requires restarting the motion process; an unconfirmed goal at shutdown requires restarting Nav2.

The checked-in ROS wheel consumer is Habitat. There is no physical ROS-to-Ventuno bridge in this checkout, and
physical stop timing has not been measured. This software stop is not a hardware e-stop.

### 3.1 Driverless-Car Trajectory Rollout & Scoring
At every control tick while navigating toward a goal or moving track:
1. Roll out **7 smooth constant-curvature / spline candidate arcs** (`"A"`..`"G"`) over a $1.5\text{ s}$ horizon.
2. Penalize occupied arcs. If every arc is blocked within 0.15 m, check tighter arcs at half speed before holding
   forward motion. In the 2D doorway regression, both off-center approaches crossed without contact.
3. Score remaining trajectories by:
   - Obstacle clearance margin
   - Progress toward target (or predicted future position `track.predict(dt)`)
   - Curvature smoothness (minimizing angular jerk)
   - **Predictive Social Yielding:** Projects every tracked person's velocity vector $(v_x, v_y)$ $2.0\text{ s}$ ahead. In narrow passages with an approaching person, penalizes blocking trajectories and rewards pulling cleanly to the right wall while slightly angling toward the passerby.
4. **Non-Stop Waypoint Pursuit:** Advances through waypoint paths using a `0.28 m` blend radius so Sprocket curves smoothly between segments without stopping (`v > 0`) until final arrival.

### 3.2 Wake-Word-Locked DOA Spinal Reflex
- `DOARingBuffer` maintains a 2.5 s sliding window of `(timestamp, audio, doa_rad)`.
- Random room chatter and TV audio are ignored.
- The instant `openWakeWord` detects *"Sprocket"*, the system resolves the DOA angle at utterance onset (from the ring buffer) and triggers a sub-100 ms turn reflex directly into the Body Controller—pivoting to face the speaker *while they are still talking* and entering the `"listening"` posture.

### 3.3 Procedural Aliveness & Expressive Chimes
Every posture is paired with non-verbal procedural audio (`chime()` in `robot/interfaces/voice.py`), reserving full TTS speech for meaningful answers and questions:
- `"listening"`: Turned toward speaker DOA + gentle damped longitudinal rock $v(t) = 0.04 \sin(2\pi \cdot 1.2 t) e^{-0.3t}$ + rising two-note `"wake"`/`"listen"` chime.
- `"curious"`: Smooth sinusoidal scan $\omega(t) = 0.35 \sin(2\pi \cdot 0.6t)$ + inflected `"curious"` chirp.
- `"happy"`: Crisp double-shimmy $\omega(t) = 0.9 \sin(2\pi \cdot 3.0t)$ for $0.6\text{ s}$ + celebratory `"happy"` trill.
- `"uh_oh"`: Smooth reverse $v = -0.15\text{ m/s}$ for $0.6\text{ s}$ + descending `"uh_oh"` tone.
- `"yielding"`: Polite $0.15\text{ m}$ back-step when a person steps inside $0.45\text{ m}$ personal space while idle/listening + soft `"yield"` chirp.

### 3.4 Proprioception, Step/Cliff Judgment & Anti-Cable Snag Reflex
- **Smart `VL53L5CX` Step vs. Cliff (`classify_tof_grid`):** Computes height delta $\Delta z = \text{expected\_floor} - \text{tof\_grid}$ across the 8×8 laser matrix:
  - $|\Delta z| \le 0.022\text{ m}$ (rugs, thresholds, tile transitions): **`"traversable"`** (maintains confidence; enables $1.35\times$ `torque_boost` when climbing step-ups $\Delta z \ge 0.008\text{ m}$).
  - $\Delta z < -0.04\text{ m}$ (stairs, ledges): **`"cliff"`** (instant hardware stop, `"uh_oh"` chime, backs up $0.15\text{ m}$, records temporary keep-out).
  - $\Delta z > 0.04\text{ m}$ (glass, wall, furniture base): **`"obstacle"`**.
- **Anti-Cable / Rug Snag Reflex (`check_proprioception`):** Compares commanded wheel drive against IMU linear/angular motion and visual flow. If `abs(v_cmd) > 0.08 m/s` and `abs(v_actual) < 0.015 m/s` for $> 0.15\text{ s}$ on flat floor, immediately cuts forward torque, backs off $0.12\text{ m}$ to release any caught cable/sock, logs a temporary blocker in Circulating Memory, and steers around.

---

## 4. Layer 1.5: Pure-Vision 360° Perception (`robot/perception/`)

0. **Who sees what.** The vision model gets the pictures and answers about them itself ("what do you see", "what
   room is this"). YOLOE-26 (large) looks only for people and for the words a tool asks for (`search("keys")`,
   `follow("the cat")`, `go_to("ball")`), and gives their boxes and masks to ranging and tracking, so the fast loops
   run without the model. There is no fixed household list: in the apartment one added about 1.4 wrong names a
   picture, and big furniture was found under 8% of the time from 22 cm anyway. In the simulator YOLOE and depth run on
   the Windows GPU (`tools/vision_server.py`, DirectML: 75 and 82 ms a frame); on the board, on its NPU.
   The detector and depth node reuse their HTTP connections, but each camera upload buffer closes as soon as its
   reply is read. The connection does not own a history of sent pixels.
1. **Metric depth from the floor (`depth.py`, the depth node in `robot/ros/perceive.py`):**
   - `Depth Anything V2 small` (ONNX) estimates relative inverse depth for each camera's newest frame.
   - `fit_to_floor()` turns it into metres: the camera's height and the IMU's tilt give the floor's true distance at
     every pixel, and a robust (RANSAC) line fit in the lower picture gives the network's scale and shift.
2. **Synthetic 360° `/scan` from where the floor ends (`synthetic_scan`, `combine`):**
   - In each image column, the first rows rising off the floor are the foot of something; it is ranged where the ray
     meets the floor (exact for anything standing on it, whatever the depth's error), an overhang's edge by its depth;
     things above the 0.30 m clearance are passed under. Each camera's sector is made at its own moment and carried
     to the newest by the odometry before the three are joined.
   - Measured in the simulator against true depth: 64% of obstacles within 1 m ranged to 15% (true depth: 92%), fake
     returns on 17% of open beams (true depth: 1%). The monocular depth, not the method, is the limit
     (docs/simulation_3d.md).
3. **3D Object & Moving Person Tracking (`locate.py`, `tracker.py`):**
   - Object depth must match the RGB camera and exposure. A complete outline whose floor-contact range agrees
     with depth refines its centre; an implausible or missing contact does not.
   - Disjoint whole objects in one image keep separate IDs even when their map estimates overlap. A much smaller
     piece within a single recently observed footprint can keep the parent's ID without shrinking it or taking
     another object's match. At a camera seam, range error along a clipped sightline does not explain a second
     object's centre lying sideways outside both footprints. The model sees one badge per track per camera.
   - Extracts oriented 3D bounding dimensions (`width`, `depth_extent`, `orientation_yaw`) from segmentation masks + depth clusters.
   - Tracks 2D map velocity $(v_x, v_y)$ via $\alpha$-$\beta$ filtering so `Track.predict(dt)` supports dynamic pursuit and social hallway yielding.
   - Associates people across cameras using clothing appearance histograms + `MobileCLIP2` visual vectors + `WeSpeaker` voiceprint IDs.
4. **Human Pose, Gaze & Pointing (`pose.py`):**
   - `estimate_pose_from_keypoints()` turns 2D skeleton keypoints + depth into 3D `HumanPose` (`facing_yaw`, `head_yaw`, `posture`: `standing`/`sitting`/`crouching`/`waving`/`pointing`, `pointing_ray`, `gaze_ray`).
   - `resolve_ray_target()` resolves *"that"*, *"over there"*, or joint-attention *"look at that"* by finding the nearest tracked object inside the ray cone or intersecting the ray with the floor ($z = 0$).
5. **Places that are not one thing (`go_toward`):** the model says the direction of the picture the place is in and
   about how far ("behind-left, 2 m"); Nav2 drives there, stopping before anything in the way. Numbered dots on the
   floor for the model to pick were tried and dropped: the 2B model took the last number nearly every time.
6. **Rooms named by the model (`RealRobot._ask_rooms`, `rooms.py`):** every few seconds, until it has five answers
   for the room it stands in, the model is asked what kind of room the front picture shows; a tour's place names win.
7. **Visual Grounding Overlay (`overlay.py`):** the tracker's ids are drawn on the pictures the agent gets.
   On an ambiguous target, the tool still refuses to move and requests the recent camera view with the most distinct candidate badges. If none are visible, it returns the choices in text alone.

---

## 5. Layer 2: Memory (`robot/agent/agent.py`, `robot/mind/memory.py`)

### 5.1 Short-Term Memory
- **The task's conversation:** everything the model saw, called and was told during the current task, with the newest four pictures kept.
- **The to-do list:** the model's own steps for a several-part request, ticked off as it goes.
- **Earlier tasks:** a one-line summary of each of the last four tasks, shown at the start of the next.

### 5.2 Long-Term Memory (SQLite; the final stack is still being decided)
Today the text "embeddings" are hashed words with a hand-written synonym table, not a learned model, and the agent reaches long-term memory only through `recall` and `remember`.

SQLite stores structured spatial/state records alongside normalized float32 embedding BLOBs (text trigram/token + synonym canonicalization, `MobileCLIP2` visual vectors, `WeSpeaker` voiceprint vectors):
- **Volatility Half-Lives (`HALF_LIFE`):**
  - `"blocker"` / `route_notes` (closed door, dropped chair, cable snag): **$1500\text{ s}$ (25 min)**.
  - `"movable"` object location (`keys`, `mug`, `phone`, `backpack`): **$7200\text{ s}$ (2 hours)**.
  - `"anchor"` furniture (`sofa`, `desk`, `fridge`, `bed`): **$604800\text{ s}$ (7 days)**.
- **Habit EMA (`anchors_habits`):** Links movable objects to nearby furniture anchors with an Exponential Moving Average ($\alpha = 0.3$). `usual_anchors("keys")` ranks the spots where keys are most often found so search checks high-habit anchors first.
- **Live Visual Overwrite ("Eyes Beat Old Memory"):** `observe_clear_space(x, y, radius)` immediately clears stale route blockers and marks vanished movable items absent when a camera inspects that space clearly.
- **Automatic Circulation (`circulate()`):** Evicts unreinforced records whose freshness $w(t) = 0.5^{\Delta t / T_{1/2}}$ drops below $0.10$.
- **Voiceprint + Appearance Fusion (`PersonProfile`):** Matches speakers by `WeSpeaker` voice vector; on match, automatically updates their `look_vec` (today's clothing) so Sprocket recognizes known people even when they change outfits.

---

## 6. Layer 2: The Agent (`robot/agent/`, `robot/mind/brain.py`)

An agentic loop, as in coding agents, around the robot's vision-language model. The design and the reasons for it are in [docs/ideas/sprocket-harness.md](ideas/sprocket-harness.md).

1. **One task is one running conversation** (`robot/agent/agent.py`). The first message holds what the robot senses — room, battery, the tracked objects (id, what, how far, which way: people and the things tools have looked for) and the three cameras' pictures with ids drawn on — then the user's words. Each model turn may speak, call tools, or both; every tool result goes back to the model before its next turn.
2. **Native tool calls** (`Brain.chat`). Tools are JSON function schemas; llama-server uses the model's chat template. Batched calls run in order. A failure or a tool returning visual evidence skips the rest of that batch: the model must see the result before it chooses its next action. A reply the server marks incomplete at its output limit dispatches no calls.
3. **Tools** (`robot/agent/tools.py`): `say`, `go_to`, `go_toward`, `go_to_side`, `stop_near`, `move`, `turn`, `drive`, `orbit`, `follow`, `face`, `keep_facing`, `search`, `look`, `look_back`, `wait`, `ask`, `remember`, `recall`, `todo`, `todo_done`. Tools that find, follow or drive to a thing pass its name to YOLOE, which then looks for that word on all three cameras while the tool's own loop runs; `go_toward(direction, metres)` is for a place that is not one thing, with Nav2 finding the way; `search(what, area, near)` asks the model at each stop which camera's way looks likeliest, and when that way has no floor to drive on (the thing may be right there, too near for YOLOE to name) steps back from it to see it whole; a target name that is neither known nor in view is searched for the same way before the tool goes on; `look_back(seconds)` shows the front camera's last pictures. A different behaviour is a different tool, not an optional argument: the 2B model filled optional arguments on almost every call (`side="front"`, `stop_distance=2` for "come here"). Targets are an object id, `user`, a room or saved place, or a description. Every call is checked first: an unknown tool, a missing or out-of-range argument, or a target that matches nothing or several things returns an error saying what to do instead (`'box' could be any of: box 1 (0.8 m front-left); box 2 … Use the id, or ask the user which one`). Nothing is silently changed.
4. **Seeing**: all three camera snapshots come with each request by default. `look(direction, what)` samples a camera again without driving, even while stationary; the optional visual name activates detection. `rear-left` and `rear-right` name the cameras directly. Marks belong to the same exposure as their pixels. A successful 3D search returns the selected track's saved crop when available, labelled as recorded evidence, and recent marked views where available. The model judges the image; a detector badge is not proof of identity.
5. **To-do list**: for a several-part request the model writes `todo([...])` and ticks steps off with `todo_done(n)`; the list is shown back with every update and on the live page.
6. **Ending a task**: a reply with no tool call is spoken and ends the task. The first time the model tries to finish after acting, a stop hook shows it the request, what it did and any open to-do steps, and asks it to check; its next plain reply ends the task.
7. **Loops**: a third identical call in a row is refused, except for `look`: objects and people can move while the robot is still. The overall turn budget still applies.
8. **Real time**: `stop` and `wait` act at once without the model. Anything else the user says while a task runs interrupts the current step and joins the conversation (`Agent.interject`), so the model can change course. Posture and chime come from events (listening at the start, uh-oh on a failure, happy on success), not from the model.
9. **Between tasks** the next request is one user turn that tells the last one step by step (each call and its result, and the reply) and older ones as one line each with their actions; the last task is not replayed as chat turns (the model card advises against multi-turn conversation). The robot keeps the place "where you were before".
10. **Within a task** it stays one tool-use turn, as LFM3-VL is trained: pictures from `look` come back inside the tool result, so the chat template keeps the task's reasoning (a new user message would drop it). Only the user speaking mid-task adds a user turn. Sampling follows the model card (temperature 0.1, top_k 50, top_p 1, min_p 0, repeat penalty 1.1).
11. **Thinking**: generation has a per-turn reasoning budget (`off`, `low`: 128 tokens, `medium`: 512, `high`: 2048). The dashboard shows complete, expandable model turns rather than 220-character fragments. The next turn after a tool result is a new generation in the same task, not a new user request.

For an explicit side and gap, use `go_to_side(target, side, distance)`; `behind` means the far side from the current approach, not the object's intrinsic back. Plain `go_to(target)` stays narrow: in a 3D control with optional fields it invented `side="right"` for "Go to the blue bicycle." Navigation rechecks the selected object's ID during the drive and cancels on withdrawal, relabelling or a move outside the original footprint. An ordinary occlusion is not a withdrawal. Parking under furniture is unsupported: the current sensors cannot certify the entire swept body's clearance.
