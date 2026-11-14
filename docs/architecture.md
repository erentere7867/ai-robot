# Sprocket v2 Architecture

**Single Source of Truth Specification — Living Body + 3-CSI Pure Vision + Circulating Hybrid Memory + Closed-Loop Visual Mind**

Sprocket is an indoor, general-purpose wheeled robot on the **Arduino VENTUNO Q**. Instead of compiling a blind text table into a static one-shot Python script behind a 2D dry-run gate, Sprocket runs a **continuous multi-rate embodied AI loop**: a 30–50 Hz driverless-car body engine, 360° pure-vision senses (no spinning lidar), a two-tier circulating hybrid memory, and a fast (~0.8 s) closed-loop Vision-Language + Calibrated Decision Mind.

---

## 1. Core Principles

1. **Living, Non-Stop Motion:** The body never freezes into warehouse-cart stop-and-go steps. It rolls out smooth trajectory curves at 30–50 Hz, blends waypoints continuously, yields politely to approaching people, and expresses attention through posture and synthesized chimes.
2. **Closed-Loop Visual Mind (See While Acting):** `LFM3-VL-2B` receives annotated camera images (Set-of-Mark object IDs + *PIVOT* projected 3D floor paths) alongside scene state and emits compact structured `Intent` updates every ~0.8 s or immediately upon semantic world events.
3. **Unified Calibrated Scoring (`d1-2B`):** Every scoring decision—*PIVOT* trajectory visual scoring, *VLFM* search frontier ranking, and Active Vision uncertainty triggers—uses `d1-2B` single-token calibrated log-probabilities (~200 ms) rather than uncalibrated chat heuristics.
4. **Pure 360° Vision + Low-Cost Multi-Zone ToF:** Three wide-angle MIPI-CSI cameras calibrated by floor-plane trigonometry, IMU pitch/roll, and 8×8 `VL53L5CX` laser ToF anchors replace the expensive spinning 2D lidar while catching low obstacles (cables, shoes, pets) below a lidar plane.
5. **Circulating Memory (Nothing Is Set in Stone):** Short-term working memory tracks active goals and recent visual frames in RAM; long-term memory decays by volatility half-life, updates movable habits via EMA, overwrites stale entries instantly when cameras see empty space, and recycles expired records.
6. **Zero Dead Fallback Legacy:** No blind one-shot script gate, no fake 2D dry-run simulation before motion, and no ungrounded guessing.

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
| **40 TOPS Hexagon NPU** | `LFM3-VL-2B` (Q4_0, closed-loop planner & describe), `d1-2B` (calibrated scorer), `YOLOE-26s-seg` (open-vocab masks) | `LFM3-VL-2B` ~0.8 s/intent, `d1-2B` ~200 ms, `YOLOE` ~12 ms/frame |
| **Adreno 623 GPU** | `Depth Anything V3 Metric-Indoor Small`, `RTMPose` / `YOLO-Pose` (keypoints/gaze/pointing), `MobileCLIP2` (crop embeddings) | 10–30 Hz total |
| **8-core Kryo CPU** | 30–50 Hz Living Body Engine, `slam_toolbox` / Nav2 costmap, `WeSpeaker` voiceprint embedding, `Whisper`, `Piper` TTS, `ChimeSynthesizer`, SQLite + Vector Memory | 30–50 Hz body, <2 ms memory lookup |
| **STM32H5 MCU** | 200 Hz wheel PID + gyro odometry, `VL53L5CX` 8×8 step/cliff governor, IMU+current anti-cable snag reflex, 300 ms command watchdog | 200 Hz–1 kHz |

---

## 3. Layer 1: The Living Body Engine (`robot/body/controller.py` @ 30–50 Hz)

The Living Body Engine runs continuously at 30–50 Hz and blends goal pursuit, obstacle avoidance, social politeness, active vision, and lifelike motion.

### 3.1 Driverless-Car Trajectory Rollout & Scoring
At every control tick while navigating toward a goal or moving track:
1. Roll out **7 smooth constant-curvature / spline candidate arcs** (`"A"`..`"G"`) over a $1.5\text{ s}$ horizon.
2. Reject or heavily penalize arcs violating occupied cells / synthetic `/scan` clearance.
3. Score remaining trajectories by:
   - Obstacle clearance margin
   - Progress toward target (or predicted future position `track.predict(dt)`)
   - Curvature smoothness (minimizing angular jerk)
   - **Predictive Social Yielding:** Projects every tracked person's velocity vector $(v_x, v_y)$ $2.0\text{ s}$ ahead. In narrow passages with an approaching person, penalizes blocking trajectories and rewards pulling cleanly to the right wall while slightly angling toward the passerby.
   - **Visual Path Prior (`d1-2B`):** In cluttered scenes or doorways, projects top candidate ribbons `(A)`, `(B)`, `(C)` onto the camera floor image (`overlay.py`) and incorporates `d1-2B` calibrated visual probabilities (~200 ms).
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

## 4. Layer 1.5: Pure-Vision 360° Perception & Active Vision (`robot/perception/`)

1. **Metric Depth Calibration (`depth.py`):**
   - `Depth Anything V3 Metric-Indoor Small` runs on all 3 CSI streams.
   - `calibrate_depth()` locks scale and offset $(a, b)$ per frame using known camera mounting height (`CAMERA_Z = 0.22 m`), live IMU `pitch_rad`/`roll_rad` floor-plane trigonometry, and `VL53L5CX` 8×8 laser distance anchors (crucial when clutter hides the floor).
2. **Synthetic 360° `/scan` (`synthetic_scan`):**
   - Back-projects all 3 calibrated depth images into robot-frame 3D points, slices obstacle heights $z \in [0.03\text{ m}, 0.38\text{ m}]$, and projects minimum ranges into a 360-beam `LaserScan` for SLAM and the 30 Hz obstacle field.
3. **3D Object & Moving Person Tracking (`locate.py`, `tracker.py`):**
   - Extracts oriented 3D bounding dimensions (`width`, `depth_extent`, `orientation_yaw`) from segmentation masks + depth clusters.
   - Tracks 2D map velocity $(v_x, v_y)$ via $\alpha$-$\beta$ filtering so `Track.predict(dt)` supports dynamic pursuit and social hallway yielding.
   - Associates people across cameras using clothing appearance histograms + `MobileCLIP2` visual vectors + `WeSpeaker` voiceprint IDs.
4. **Human Pose, Gaze & Pointing (`pose.py`):**
   - `estimate_pose_from_keypoints()` turns 2D skeleton keypoints + depth into 3D `HumanPose` (`facing_yaw`, `head_yaw`, `posture`: `standing`/`sitting`/`crouching`/`waving`/`pointing`, `pointing_ray`, `gaze_ray`).
   - `resolve_ray_target()` resolves *"that"*, *"over there"*, or joint-attention *"look at that"* by finding the nearest tracked object inside the ray cone or intersecting the ray with the floor ($z = 0$).
5. **Active Vision (`needs_active_vision` & `BodyController`):**
   - Triggered when `d1-2B` returns uncertain confidence ($0.30 \le P \le 0.80$), an object is edge-cut/occluded, or small lost items are sought near furniture:
     - `peek`: Arcs $0.35\text{ m}$ laterally to see past an occlusion.
     - `step_closer`: Closes distance for higher pixel density.
     - `step_back_under`: Steps back to a $1.25\text{ m}$ stand-off from a furniture anchor (`sofa`, `bed`, `desk`) so cameras angle *underneath* it.
6. **Visual Grounding Overlay (`overlay.py`):**
   - Draws high-contrast Set-of-Mark object IDs (`[1]`, `[2]`) and projects *PIVOT*-style 3D floor candidate trajectories (`(A)`, `(B)`, `(C)`) directly onto the camera image sent to `LFM3-VL-2B` and `d1-2B`.

---

## 5. Layer 2: Two-Tier Circulating Hybrid Memory (`robot/mind/memory.py`)

### 5.1 Short-Term Working Memory (In RAM)
- **`TaskScratchpad`:** Tracks `goal`, `subgoal`, `visited_rooms`, `inspected_ids`, `checked_spots`, `blockers`, and `steps`. Prevents search loops and keeps subgoals coherent across steps.
- **`VisualRingBuffer`:** Holds up to 16 recent keyframes across 15 seconds (`(timestamp, camera, pose, rgb, track_ids)`), enabling *"what did you just pass?"* lookbacks.
- **`SpeakerLock` & Dialogue:** Active speaker `PersonProfile` ID, DOA angle, and recent conversation turns.

### 5.2 Long-Term Circulating Memory (SQLite + Embedded Vector Index)
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

## 6. Layer 2: Closed-Loop Visual Mind (`robot/mind/brain.py`, `robot/agent/`)

1. **Instant Reflex Commands (`robot/agent/instant.py`):** `"stop"`, `"wait"`, and `"come back"` act immediately without waiting for any language model.
2. **Closed-Loop `Intent` Planning (`robot/agent/planner.py`, `robot/agent/agent.py`):**
   - At ~0.8 s cadence (or immediately when semantic world events fire: target moved, target lost, occlusion/blocker detected, new object matching goal spotted), `LocalPlanner` sends `LFM3-VL-2B` the **annotated Set-of-Mark + *PIVOT* camera frame**, the **scene summary** (with velocities, human pose/pointing, scratchpad, habit anchors, and semantic facts), and receives a compact grammar-constrained `Intent`:
     - `say` (spoken update if appropriate), `chime` (`wake`/`listen`/`curious`/`happy`/`uh_oh`/`yield`), `posture`, `action` (`go_to`, `pursue`, `orbit`, `search_frontier`, `peek`, `step_closer`, `look_under`, `face`, `move`, `turn`, `spin`, `motion_sequence`, `ask`, `remember`, `done`, `refuse`), `target`, `params`, `subgoal`.
   - The Body Controller starts executing immediately without blocking future visual re-evaluation.
3. **Unified `d1-2B` Decision Scoring (`Brain.decide`, `score_paths`, `score_frontiers`):**
   - `d1-2B` outputs calibrated probabilities $P \in [0, 1]$ in ~200 ms for:
     - **Trajectory tie-breaking:** Scoring projected floor paths `(A)`, `(B)`, `(C)`.
     - **VLFM Frontier Search:** Ranking unexplored doorways/views by likelihood of holding the search target.
     - **Active Vision Gate:** $P > 0.80 \rightarrow$ confirm; $0.30 \le P \le 0.80 \rightarrow$ trigger `peek` / `step_closer`; $P < 0.30 \rightarrow$ reject.
4. **Escalation & Honesty:** If the local planner fails twice on a step and an optional cloud/stronger escalation planner is configured, it gets one attempt; otherwise Sprocket plays `"uh_oh"`, states plainly why it cannot proceed, and asks the user for help without guessing.
