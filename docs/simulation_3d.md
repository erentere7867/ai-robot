# Sprocket v2 — 3D Simulation

One simulator, and it stands in for exactly what the real robot gets from its hardware: the world, the body, and the
raw sensors. Everything above that is the robot's own software, unchanged. Nothing of the simulator's truth reaches
the robot; the judge alone reads it.

## What is simulated, and what is not

| Layer | In the simulator | On the robot |
| :--- | :--- | :--- |
| World and body | `robot/sim/habitat_server.py`: a ReplicaCAD apartment in Habitat-Sim; a 0.30 m round body that rolls on the floor and stops where it would touch anything (no sliding along walls); every piece of furniture solid | the home |
| Drivers | `robot/sim/habitat_ros.py`: three 1280×720 140° **colour** cameras at 5 Hz (no depth), front/rear 8×8 ToF grids at 10 Hz, wheel odometry and a gyro at 50 Hz, a motor controller with acceleration limits and a 0.3 s command watchdog; it owns `/clock` | camera drivers, the MCU, the IMU |
| Everything else | the robot's own code: robot_localization EKF (wheels + gyro), the depth node, slam_toolbox or AMCL, Nav2, YOLOE-26, the tracker, the agent | the same |

Sensor errors are modelled where the real parts have them: wheels misjudge distance by about 1% and report 5% too
much turning on the spot (tyres scrub), the gyro has a fixed bias, the ToF zones are noisy. The judge gets
`/ground_truth` (where the body really is), a bump count, and the apartment's objects; the robot never does.

Simulated time runs at `--rtf` times wall time (default 0.33). On this PC the robot's software runs on the CPU (YOLOE
about 0.4 s a frame, the depth network about 0.7 s), far slower than on the board's NPU and GPU; slowing the world
gives it roughly the board's share of work per simulated second. Model calls take wall time, but nothing moves while
the agent waits for one.

## Seeing obstacles without a lidar

The robot has no depth camera and no lidar. Its obstacle scan comes from the colour cameras:

1. **Depth Anything V2 small** (ONNX, `models/depth-anything-v2-small-onnx/model.onnx`) estimates relative
   inverse depth for each frame (`robot/perception/depth.py`, `MonoDepth`).
2. **Metres from the floor's geometry** (`fit_to_floor`): the camera's height and the IMU's tilt give the floor's
   true distance at every pixel; a robust line fit of the network's output to it in the lower picture gives the scale
   and shift. Against true depth, floor pixels come out within 15% for 96% of them.
3. **Where the floor ends, per image column** (`synthetic_scan`). Down a column the floor's inverse depth grows row
   by row at a rate the camera's height and tilt fix; up the face of anything standing on the floor it stays nearly
   constant. The network's depth is often wrong by a scale over a whole surface (a wall put twice as far as it is),
   but the surface still keeps a constant inverse depth. So the foot of something is found from the **rate**, not
   the depth's value: the lowest row where inverse depth grows at under half the floor's rate, for 3 rows and 6 cm of
   height (a rug's edge is floor). It is ranged where its ray meets the floor; an edge clearly above the floor (a
   sofa's front over a gap, a shelf) is ranged by its depth; something at the robot's height clearly nearer than
   the face behind it (depth against depth: a table's edge, the stairs' underside) by its depth; things above the
   robot's 0.30 m clearance are passed under. Beams: the nearest return; inf where the floor was seen clear; nan
   where nothing was seen.
4. **One 360° scan from three cameras taken at different moments** (`combine` in `robot/ros/perceive.py`): each
   camera's sector is made at its own picture's moment and carried to the newest one by the odometry.

Measured against the ideal scan from true depth (20 random spots in `apt_0`, obstacles 0.3 m or more away; "too
far" means beyond the real obstacle by over 15%, which tells the map builder the floor is clear through it):

| Depth, scan | within 15%: 0.3–1 m | 1–1.5 m | 1.5–2 m | 2–3 m | 3–8 m | too far | fake returns on open beams |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| true depth, foot by rate (the method's limit) | 91% | 97% | 95% | 85% | 75% | 3% | 5% |
| Depth Anything V2 small, foot by rate | 83% | 77% | 73% | 69% | 49% | 9% | 17% |
| Depth Anything V2 small, foot by height (before) | 77% | 55% | 51% | 45% | 24% | 15% | 21% |
| yolo26n-depth (before that) | — | — | — | — | — | — | a ring of fake obstacles 0.3–0.5 m all round |

The earlier version found the foot where depth put a point 6 cm above the floor. That is exact only when the depth
is: a wall the network puts 30% too far is "floor" 30% further up its face, the scan said it was 30% further away,
and the map builder cleared floor through it. The larger network input (518 px) and the metric-indoor model were
worse on this set, not better. Cut into narrower virtual views the network did better at mid range but either
missed the floor at the robot's feet (level views) or added fake returns (tipped views).

## What YOLOE finds

The robot turns once, slowly, on its dock in `apt_0`; its tracks are then matched with the objects that were plainly
in view at some point (at least 300 pixels of the judge's 640x360 semantic picture), by name and within a metre of
where they are:

| Vocabulary | objects plainly in view | found, right name and place | tracks with no such object |
| :--- | :--- | :--- | :--- |
| 42 household words | 37 | 2 (a chair, the sofa) | 9 (clocks, shelves, a ball) |
| 54 (+ bicycle, stool, rug, monitor, bean bag, chest of drawers, kitchen counter, sink, oven, microwave, mirror, stairs) | 38 | 4 (+ two beanbags) | 6 |

On single pictures it does find chairs (0.78), the sofa, the monitor (as "tv", 0.45); it calls a bike's spoked wheel
a clock (0.87, even with "bicycle" in its words), the kitchen cupboards shelves, and has no name at all for a desk or
a table seen from 22 cm below its top: a dark slab on legs. Half of each 140-degree picture is floor; the furniture
sits in a thin band across the middle. Words a request names are added at run time (`Detector.watch`), so "go to the
bike" looks for a bike; this table is what the robot notices without being asked. A text list of detections alone
would leave the model nearly blind here: the picture carries most of what it knows.

### Bigger YOLOE, one word at a time, and the vision model (19 November)

40 front-camera pictures from random spots in `apt_0`, each with the truth of what is in it. Per word, how often a
thing plainly in view (300+ px) is said to be there ("found"), and how often it is said to be there with none of it
in view at all ("false"); YOLOE at the robot's threshold, 0.22:

| | found | false | false names per picture, full list | ms a picture, RX 6600 (DirectML) |
| :--- | :--- | :--- | :--- | :--- |
| YOLOE-26s, 53-word list | 20% | 2% | 1.2 | 29 |
| YOLOE-26m, 53-word list | 19% | 2% | 0.8 | 67 |
| YOLOE-26l, 53-word list | 23% | 4% | 1.4 | 75 |
| YOLOE-26l, asked for that word alone | 27% | 4% | — | 75 |
| LFM3-VL-2B, "Is there a ... in this picture?" (`Brain.check`) | 79% | 19% | — | 3.8 s for 19 questions |
| both say yes | 24% | 3% | — | |

- **Bigger YOLOE sees object-sized things much better**: from s to l, plants 35% → 76%, chairs 31% → 62%, TVs
  40% → 73%. It does not see furniture from 22 cm at any size: tables, kitchen counters, rugs, the fridge 0–8%, even
  when they fill a good part of the picture (`docs/yoloe_large_22cm.jpg`: the fridge is a black wall, the back of
  the sofa a "mirror" at 0.74). The vision model finds them: tables 96%, counters 90%, rugs 75%.
- **Asking YOLOE for one word does not make it see better.** Each word is scored on its own (a sigmoid of the
  picture's features against that word's text embedding), so a word's score is the same alone or in a list. What
  changes is that the list keeps each box under its best word only: asked alone, the large model found chairs 75%
  (62% in the list), cabinets 35% (16%), bikes 50% (25%). A word the list had a better synonym for did worse alone
  ("plant" 71% against "potted plant" in the list); for the small and medium models asking alone was slightly worse
  overall (18%). False alarms for the asked word stay: "ball" in 12% of the ball-free pictures with the large model.
  Over all 19 words the larger models gain little (20% → 23%) because big furniture, which no size sees, is most of
  what is in view.
- **The vision model says yes too easily**: a ball in 32% of the pictures that have none, a chest of drawers in 64%.
- The exported YOLOE on the GPU gives the same boxes and scores as ultralytics on the CPU (168 of 168 at 0.22).
  Depth Anything V2 small takes 82 ms on the same GPU with its input size fixed (231 ms with it open, about 600 ms
  on the CPU).
- **The vision model is not fast enough to steer**, on the RX 6600 with nothing else running (llama-server, Vulkan):
  a 512-wide picture is about 146 tokens and takes about 150 ms to read. 8 frames (4 s of video at the model's 2
  frames a second) take 1.2 s before the first word; a yes/no on one picture takes 0.2 s; answers come at about 136
  tokens a second, so a decision that thinks up to its 512-token cap adds up to about 4 s. The model card's video
  input is a whole clip sampled at 2 frames a second, with no streaming mode. The board will be slower.

### What the harness learned from running it (19 November, apt_0)

- **The fridge is no test.** ReplicaCAD draws it as a flat panel that reads as a cabinet; YOLOE-26l scored it 0 for
  "fridge" and "refrigerator" from 22 cm and from 1 m alike (docs/fridge_22cm_vs_1m.jpg). The fridge tasks now use
  the TV.
- **"table" is a bad word for YOLOE, and height still matters.** Asked alone, the best "table" or "dining table"
  score in pictures of tables from 1 m was 0.03-0.14, where "desk" and "coffee table" scored 0.25-0.69 on the same
  pictures. Ultralytics' own code and the GPU server gave identical scores, so it is the model. Even "desk" found
  them 54% of the time from 1 m and 8% from 22 cm (docs/table_22cm_vs_1m.jpg).
- **Picking numbered dots does not work with the 2B model.** Asked which of 20-odd dots across the three pictures to
  search towards, it took the last number nearly every time ("dot 23 of 23", "dot 4 of 4"), with reasons like
  "most likely at Dot July". A yes/no per camera ("would going this way lead to a plant?") gives real choices
  (front 22%, rear-left 90%, rear-right 11%), and the search now drives the likeliest way.
- **YOLOE fires on the asked word too easily** (black panels and the black ceiling as "tv"): three TVs were known in
  a flat with one, and the robot drove to a false one. The vision model's no on the crop now drops a detection, and
  the robot waits for that check before driving to something newly found; the next run knew one TV and passed.
- **Places that are not one thing go by direction, not dots.** Asked which numbered floor dot in the front picture
  was "between the sofa and the table", the model picked the last dot (7 of 7, 7 of 8) or one with no path; when
  the two were behind it, no dot could be right. go_toward takes the direction of the picture and a distance; its
  goal is a point the planner can reach (round what is in between), not merely free floor, which had answered
  "blocked" to most calls.
- **A search keeps away from all of its own trail**, not only its stops: it went back and forth along one path.
- **A thing is gone only after six clear misses and 3 s unseen.** Misses come from all three cameras; two of them
  missing a flickering plant took it off the map a second after a search found it.
- **Still open:** the 2B model picks the wrong tool more often as tools are added (go_toward for "go to the sofa"),
  and YOLOE's false sofas and TVs get past its yes/no check on the crop (six "sofas" known in a flat with one).
- **Stalls of 20-35 s** in answers from the GPU server were the connection, not the GPU: /health, which never waits
  for the GPU, took 19.5 and 35.6 s. A new connection per request at about 16 a second from WSL; with connections
  kept open (HTTP/1.1), an hour of probing every half second across six 3D runs, and a load test, saw none.
- **Keep the connection, not the pixels.** Completed HTTP requests could retain full frames until Python's cyclic
  collector ran. Streaming the upload from a buffer closed after each exchange reduced peak process memory from
  305-306 to 58-59 MiB in three paired 120-frame loopback runs (720p, no inference). On eight apartment pictures
  repeated three times, the DirectML client's detection arrays and depth pixels matched byte for byte in all 24
  pairs. Median detector/depth exchanges were 113/72 ms before, 111/70 ms after. These are PC transport measurements,
  not board latency or a bound on whole-robot memory.

## First time in a home

`robot/sim/stack.py` `Home(scene)`: if `maps/<scene>.yaml` does not exist, the robot explores (frontier
exploration on Nav2, `robot/ros/explore.py`) while slam_toolbox builds the map, and saves it; its map's frame starts
at its dock. Afterwards it localises in that map with AMCL from the dock. `Home.robot(detector, brain)` gives a
RealRobot that knows only its own map: rooms are segmented from it and named by the furniture it sees.

In `apt_0` (19 November) exploring ended by itself after 347 simulated seconds (11 frontiers reached, 5 given up, none
left). The map shows 94% of the reachable floor as free; 20% of what it shows free is not floor the body can reach,
most of it three pieces of furniture the cameras see under or past (a coffee table on a tube frame, a desk, a kitchen
cabinet on a rug). Driving to 12 random spots on it afterwards:

| | reached within 0.5 m | localisation off, median / worst | contact steps |
| :--- | :--- | :--- | :--- |
| first | 6 of 12 | 0.37 m / 0.63 m | 3176 (minutes against the coffee table's frame) |
| with the stall check (below) and AMCL told the scan's measured noise | 9 of 12 | 0.26 m / 0.61 m | 30 |

AMCL's defaults (`bringup.AMCL`) are for a lidar: a 0.1 m hit and 10% random returns. The camera scan has about a
quarter of its returns nearer or further than anything there, so 0.2 m and 30%, and 60 beams rather than 150 (one
misjudged wall spoils many beams at once). Far from the dock (the living room's far end, 5-6 m away) it is still
about 0.5 m off.

**Touching what no sensor saw.** When the motors are driven (the command after Nav2's collision monitor) and the
wheels do not turn for 0.3 s, something holds the body: the depth node puts that spot, at the body's edge, into
/scan for two minutes (`SyntheticScanPublisher.wheels`), so Nav2's costmaps mark it and the planner goes round,
where before it recovered and drove into it again. Without a bumper this rests on the wheels stalling; wheels that
slip on a hard floor would hide the contact.

## Running

Once: `wsl -- bash tools/linux/setup_habitat.sh`, `tools/linux/setup_models.sh`, and the depth model above. Every
shell: `. tools/linux/env.sh` (ROS, the Python env, OpenGL on the GPU, and ROS over shared memory: with WSL's mirrored
networking, DDS over UDP finds the other processes but delivers no data).

```bash
python -m robot.eval.home3d --tag first            # 27 tasks, judged against the truth
python -m robot.interfaces.demo                    # talk to it on http://127.0.0.1:8765
bash tools/linux/stop.sh                             # stop everything a run left behind
```

## Fixed on the way here

- The Gazebo setup gave the cameras perfect depth images; the real cameras have none.
- Habitat's navigation mesh floats 12 cm above the floor; the old 3D sim put the cameras there (0.34 m, not 0.22 m).
- ReplicaCAD loads 102 of 113 objects and every articulated one (fridge, counters, cabinets) as dynamic, and the
  navigation mesh leaves those out: the body could drive through 36 large pieces of furniture.
- The old 3D sim "detected" objects from the simulator's own labels, and drove with its own path follower.
- On Habitat's default 5 cm grid the body's mesh kept its centre 0.21 m (median) from surfaces, not its 0.15 m
  radius: it "touched" things 6 cm early. Built on a 2.5 cm grid it is about right.
- Nav2 dropped the scan's "floor seen clear" (inf) readings by default, so a wrong reading could never be cleared;
  slam_toolbox made a cell wall when 10% of beams through it ended there (right for a lidar, not for estimated
  depth: now 30%); the ToF sensors never reached Nav2, so what the cameras take for floor (the lowest 6 cm) was
  invisible to it. All three fixed in the robot's own software, not the simulator.

## Hazards found

- **Under the stairs.** Exploring `apt_0`, the robot drove under the open staircase, whose underside slopes down
  below its height, and stayed wedged for 2.5 simulated minutes while Nav2 kept trying. A low overhang near the
  robot is above the cameras' horizon and thin in the picture; the scan now trusts depth for anything at the
  robot's height in front of the floor's edge (within 2 m), but monocular depth is weakest exactly there. Robot
  vacuums get stuck the same way, and carry a bumper and an upward sensor for it: a microswitch bumper round the top
  deck, or a third VL53L1X-class ToF tipped up and forward, would cost a few dollars and close this for good.
- **Tube-frame furniture.** The living room's coffee table stands on a sled of 2 cm tubes that rise diagonally to
  its top. The depth network smooths them away, the ToF zones are far wider than a tube, and the map shows floor
  under the table: the robot drove into the frame from three sides (1766, 621 and 171 contact steps). Chair legs,
  bar stools and clothes racks are the same. The stall check above stops the repeating; a bumper would also catch
  the contact when the wheels slip instead of stalling.

Observations from the earlier sim that still hold: from 0.22 m the cameras never see the top of an 86 cm counter or
a sofa's seat; "what is on the table?" needs a taller mast or a camera that tilts up.
