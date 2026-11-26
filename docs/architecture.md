# Sprocket v2 Architecture

Sprocket is an indoor, general-purpose wheeled robot designed for the **Arduino VENTUNO Q**. The vision-language
model sees the cameras and decides what to do through native tool calls. The tools handle sensing and motion and
return evidence before the model chooses its next action.

This document describes the checked-in stack. Habitat supplies the body and raw sensors for the ROS stack; it
does not supply perception, navigation, or decisions. The STM32 sketch is separate: this checkout has no physical
ROS-to-Ventuno wheel bridge. See [simulation_3d.md](simulation_3d.md) for measurements and [usage.md](usage.md) to run it.

---

## 1. Core Principles

- The model decides; the harness validates arguments, reports the outcome, and stops motion when necessary.
  Exact stop/wait commands are handled without the model.
- Camera and ToF readings make a synthetic scan, not a simulator-truth view. Nav2 uses the scan and the robot's
  own map. YOLOE looks for people and the terms requested by tools rather than a fixed household vocabulary.
- Control loops run independently of model generation. Evidence, task progress and remembered facts are available
  to the agent; a detector badge or commanded motion alone is not proof of success.

---

## 2. Hardware and development stack (Arduino VENTUNO Q)

The target is the VENTUNO Q with 16 GB shared RAM and an STM32H5. The robot model is a 0.30 m differential-drive
base with three 140° cameras, mounted 0.22 m above the floor and facing 0°, +120°, and −120°. Habitat also
provides front and rear 8×8 ToF grids, wheel odometry, and a gyro. On the development PC the VLM and the
YOLOE-26l/Depth Anything V2 Base FP32 vision server run on the Windows GPU; ROS, the agent and Habitat run in WSL.
See [testing.md](testing.md) for the Linux setup. Throughput and memory on the target board are unmeasured.

**STM32 motor safety** (`firmware/ventuno_mcu/ventuno_mcu.ino`): the sketch runs 200 Hz wheel control. E-stop,
bumper, and unhealthy ToF can cut motor standby; a Zephyr timer cuts standby after 300 ms without commands
or 50 ms without a completed drive update even when the sketch loop stalls. These behaviours are host-tested;
target build, hardware wiring and physical cutoff timing are unverified. The timer is not a substitute for a
wired e-stop.

On the VENTUNO Q, wire the cliff inputs in sketch order to D25, D26, D13 and D27, and the head servo to D28
(D25–D28 are JMISC pins 1–4). To re-arm after a latched stop, stop sending velocity commands, release the
e-stop and clear the hazard. If software e-stop is set, call the Router Bridge RPC `estop(false)`. Then call
`reset_faults()` and wait for `true` before sending a new velocity command; `false` means a safety input or
the ToF is still unhealthy.

---

## 3. Body and navigation (`robot/body/`, `robot/ros/`)

**ROS and Habitat.** Nav2 follows paths; direct skills (such as gestures) send their own commands. Both use
`MotionGate` (`robot/ros/motion.py`): only the active owner can send commands through Collision Monitor, and
only fresh matching replies reach `/sprocket/cmd_vel_wheels`. The base limits acceleration. Neither Nav2's
default `/cmd_vel` publishers nor a late reply after cancellation can bypass the gate. Stop closes the gate
and sends zero before requesting cancellation; new motion waits for a terminal Nav2 result. A ROS clock rewind
requires restarting the motion process; an unconfirmed goal at shutdown requires restarting Nav2.

`MotionGate.expire` checks at 40 Hz and sends zero after 250 ms of source age; Habitat independently stops
on stale wheel commands using the same age limit. Keep the ROS stop independent of the MCU watchdog: the latter
is unverified on hardware. This software stop is not a hardware e-stop.

**Navigation and search.** `RosRobot` checks a goal selected on the saved map against Nav2's live costmap:
`ComputePathToPose` must return a path ending within 0.10 m of the goal. `SimRobot.search` tries the next
viewpoint or ranked frontier after a rejection, without a per-round candidate cap, for at most 20 rounds.
A failed stop that caused no motion skips the camera wait; search takes an extra final look only if its
last attempted stop failed without motion. The progress
guard applies only to Nav2's `NavigateToPose`, not `FollowPath`, `Spin` or `DriveOnHeading`. It credits
reaching ground at least 0.25 m from earlier credited positions and stops after 30 simulated seconds
without progress. Detours count; stationary replans and oscillation do not. Following
commands zero if there is no route and retries at most twice per simulated second.

**Frontier selection.** Initial SLAM mapping and task-time search share a grid-based frontier extractor:
it groups free cells at the edge of unknown or unseen space. For search in a mapped home, `RealRobot`
uses Nav2's full global costmap and applies incremental updates to it. It masks inscribed and lethal cells,
plans short A* hops and ranks them by path length. The model only reranks those candidates, scoring cameras
that face them when candidates span two or more camera directions. If no score reaches 0.5, search keeps
shortest-path-first order. Reranking does not discard any candidates; Nav2 still checks reachability.

**Mapping.** For an unmapped layout, `Home` starts slam_toolbox and `Explorer.explore` drives to frontiers,
waits for refreshed maps before declaring completion, and moves on from unreachable goals. The saved map is
used for AMCL localization on later runs. This initial mapping loop is separate from task-time search.

**2D simulation.** `BodyController` scores motion arcs and blends clear waypoints. When normal-speed arcs are
blocked it tries tighter half-speed arcs and holds forward motion if none is safe. 2D pursuit shares the progress
guard, also crediting waypoint advances. The 2D rollout is not the ROS path follower.

---

## 4. Perception (`robot/perception/`, `robot/ros/perceive.py`)

- **Detection and tracking.** YOLOE-26l detects people and the words tools ask for. Range and tracking use
  detections from the same camera and exposure. Stable IDs are drawn on the matching pictures; the model can
  answer about the pictures directly, even when YOLOE has no label. Room segmentation uses the robot's map;
  vision can propose room names, while named tour places take precedence.
- **Depth and scan.** Depth Anything V2 Base FP32 runs on the vision server, including on CPU-only hosts.
  The ROS depth node uses `fit_to_floor` to turn its relative inverse depth into metres using camera geometry
  and tilt. Each camera scan ranges floor edges and uses depth for overhangs;
  odometry compensates for different exposure times when joining the sectors. Nav2 receives the fused `/scan`
  with ToF and contact-based obstacles. The agent's four range sectors come from the contact-free
  `/scan_measured` and label stale or missing readings rather than calling them clear.
- **Grounding.** When the VLM grounds a target in an image, two adjacent positive thirds can share a crop. With
  three positives it refines the strongest first, falling back to the merged crop if refinement fails. A
  scan-backed estimate is placed on the beam that returned, offset by the estimated object radius. A badge
  or detection is evidence, not proof of identity.
- **Hot paths.** Depth uses bounded ray caches and vectorized floor fitting and scan conversion. Image resizing,
  room segmentation and vision decoding avoid repeated work; detector and depth requests reuse connections,
  while upload buffers close after each response. See the dated [simulation results](simulation_3d.md).

---

## 5. Memory (`robot/agent/agent.py`, `robot/mind/memory.py`)

- **Working context.** The current task keeps tool calls and results, a model-managed to-do list, and up to
  three recent images. The next request contains the last task step by step and older requests as summaries,
  not a replay of their chat turns.
- **Durable memory.** `Home.robot` opens a map-specific `<scene>.memory.db` by default. `RealRobot` restores
  tracks and saves tracks, notes and places; `remember` writes facts and `recall` searches for them. Matching
  stored facts also enter the first model prompt; when facts conflict, the prompt labels later facts as
  superseding earlier ones. Text matching uses hashed word features, not a learned embedding model.
- **Evaluation.** `home3d` uses an isolated `memory.db` per task. `R1R` checks recall of a corrected fact after
  restarting the robot and agent with the same database; `--no-memory` disables the store for comparison.

---

## 6. Layer 2: The Agent (`robot/agent/`, `robot/mind/brain.py`)

An agentic loop, as in coding agents, around the robot's vision-language model. The design and the reasons for it are in [docs/ideas/sprocket-harness.md](ideas/sprocket-harness.md).

1. **Observe.** The first prompt includes the room, available pose and localization uncertainty, recent range
   sectors, tracks and their distances, and by default all three camera views. Captions identify badges in
   each exposure; last-known tracks are listed separately. Missing pictures and readings are labeled.
2. **Act through native tool calls.** `Brain.chat` uses the model's chat template and JSON tool schemas.
   `Tools.execute` validates calls and returns errors that explain what to send. Motion reports measured
   displacement or turn and distinguishes an action that ended short from success. A truncated model reply
   runs no tools. Within a batch, a failure, clarification, search result or requested visual update stops
   the batch so the model can act on the result. Pictures come back inside the tool result.
3. **Find and navigate.** Tools include `go_to`, `go_to_side`, `go_toward`, `follow`, `search`, `look` and
   `look_back` (schemas in `robot/agent/tools.py`). An ambiguous target returns candidates, with a picture if
   it can show them distinctly, instead of moving. Search reports whether the requested landmark was reached
   and whether a candidate was found; it can return a recorded crop and recent marked views. `go_toward`
   takes a camera direction and approximate range; `go_to_side` uses the current approach for “behind.”
   Navigation rechecks object IDs during motion; ordinary occlusion is not a withdrawal. Parking under
   furniture is unsupported.
4. **Plan and finish.** `todo` revises the plan without losing completed steps; `todo_done` and `todo_blocked`
   distinguish completion from a blocker. Read new step numbers before marking them after a revision.
   Repeated calls and the overall turn budget are guarded. An empty reply gets one repair turn. A plain reply
   can finish the task after a review of actions and plan state; open steps are reported, not silently passed.
   Unresolved partial motion triggers a review before the final reply if another turn is available.
5. **Interrupt.** Stop and wait bypass the model. Other words during a task stop the current action and join
   the conversation with fresh observations. The loop discards superseded tool calls and stale plain or
   truncated replies rather than acting on them. The dashboard exposes the tool results and model turns.
