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
much turning on the spot (tyres scrub), the gyro has a fixed bias, the ToF zones are noisy. Like the MCU, the
simulated driver estimates gyro bias from readings while the motors are uncommanded and both encoders are still.
Without calibration a parked robot's heading drifted 53 degrees in a 220-second replay, enough to put a visible
target inside a mapped obstacle. Calibration reduced that to 0.5 degrees on the same readings. The judge gets
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
  pairs. Median detector/depth exchanges were 113/72 ms before, 111/70 ms after. In two bounded apartment searches,
  the parent peaked at 4,340 MiB evaluator RSS with up to 2,509 MiB of cached uploads (2,791 frames sampled);
  the change at 2,008 MiB with no cached uploads (4,756 frames). Both finished, on different routes. This is not
  a task-speed comparison, board latency, or a bound on whole-robot memory.

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

### Watched approaches (20 November, apt_0)

Asked to go to the nearest chair and describe the view, the model first got two new track IDs both listed as
"1.7 m right", but no picture with those IDs. Returning one marked camera on ambiguity exposed the next fault:
two separate chairs both labelled `1 CHAIR` after driving. In a nine-second drive replay, two shadow trackers
received identical detections: the old tracker gave one ID to 17/31 disjoint pairs, the new one to 0/31.
The running tracker also had 0/159 on a model-driven approach. These are repeated sightings, not 159 objects.
On 144 sparse recorded views replayed 75 times, median tracker CPU per frame (per-route averages) was 4.8 vs
6.6 microseconds. Distinct IDs do not guarantee stable identity: chairs 1 and 2 became 7 and 2 across a turn.

The three model-driven approaches ended 0.67/0.69/0.68 m from the nearest chair's edge, without contact.
All picked the slightly farther chair in the initial view (matched against the judge's dock pixels offline);
the last two spoke a description but hit the 75 s budget. No improvement in nearest-object navigation established.
A separate turn-and-count request issued no motion calls and spoke an unfinished answer; the two 912-token
responses exhausted their generation budgets. The stationary control had 0/225 shared IDs in either shadow.
Pictures, transcripts and reports: `runs/watched-1120/{before,after,distinct,turn,replay}`.

#### Wide-angle handoff (`runs/handoff-depth-01`)

- On a 21 s turn/drive/turn, six of 58 frames that placed an object from depth used another exposure. In one,
  the robot had turned 8.9 degrees: a chair centre shifted 40 px in a box only 72 px wide. Object ranging now
  matches camera and capture stamp, waiting at most 150 ms before using the existing floor/scan path. In the
  20 s repeat 77/101 processed frames had matching depth; none used another exposure. Both drives had zero bumps.
- At the dock the farther chair spanned 181 px and the nearer one 88 px, but their **angles** were 16.6 and
  15.8 degrees. The pinhole correction already accounts for this. Across 23 off-axis chair sightings (>48 degrees),
  median absolute centre-range error was 0.23 m with learned depth vs 0.03 m substituting judge depth offline.
- YOLOE sometimes boxed two legs of **one** chair separately. On 16 re-rendered views, cylindrical input removed
  both duplicate detections but missed a chair on another frame. On eight other saved plant views it found
  four genuine instances instead of three but returned three unmatched boxes instead of one. No remap shipped.
  The N3 model run stopped 0.67 m from the nearest edge with no bump; proximity still does not establish identity.
- Remapping input to the same depth network also had mixed results. On 42 saved object masks the median
  per-mask surface-range error fell from 0.46 to 0.30 m, but on 23 masks from this turn it rose from 0.14 to
  0.15 m. This was an offline surface check, not a test of obstacle clearance; the depth input is unchanged.

#### Grounding follow-up (`runs/shape-handoff-01`)

- The dock chairs' real centre ranges were 1.81 and 1.95 m. Learned depth alone gave 1.68 and 1.71 m.
  Combining a compatible mask-foot range with depth gave 1.79 and 1.91 m on the same boxes. This leaves an
  elevated outline alone if floor contact is implausible; it does not correct every monocular depth error.
- On two saved turns, 7 of 155 judge-identified chair sightings had an ID other than the original ID for that
  chair. Treating a narrow detected part as part of an established footprint reduced that to 0/155. The separate
  live 21 s turn had 0/65 and no contact. These are repeat observations of **two** chairs, not independent objects.
- Starting turned 30 degrees put a whole chair in front and the other chair's leg in a rear camera. Four
  replays (both camera orders, matched depth or intermittent depth) collapsed them to one ID without the
  sightline check and kept two with it. A later live drive also exposed a *late* merge: keeping the distinction
  only for the current frame is insufficient. The shared tracker now retains the cross-camera separation.
- The model can choose the wrong ID from numerically correct ranges, treat "the nearest chair" as an object
  class, or drive by a guessed direction without looking up either object. Ranked results and marked images
  cover visible candidates; visual-name validation refuses distance comparisons rather than moving. In two
  initial dock runs it chose the correct physical instance and stopped 0.59/0.57 m from its edge. A later
  dock repeat instead drove in a guessed direction and timed out. The proximity score alone cannot test this.
- Final stock N3 at the dock and at -30 degrees: ID 2 matched the nearest physical instance (38) on departure
  in the rear-right camera **and** on arrival in front. Final box IoUs were 0.97 and 0.95; endpoints were 0.57
  and 0.74 m from its edge with zero bumps. Both replies nevertheless implied the *user* had moved, too.
  These two runs verify this handoff, not model reliability across homes.
- Hardware sanity check, **not a lidar benchmark**: sampling the judge's horizontal pixel row from each
  camera origin at one-degree intervals intersected a chair in all 22 visible chair/view pairs across 16 saved
  views. Only 2 and 3 beams hit the respective chairs at the dock. This says nothing about reflectance, noise,
  a different mounting height, or obstacles outside that plane.

#### Visible object described by colour (`runs/blue-grounding-SuyM`)

An interactive request for a blue bicycle activated YOLOE for "bicycle", but the tracker stored "bike" and its
box centre read brown. Replaying the dock views with YOLOE's masks gave black, not blue: tyres and background
outnumber the frame pixels. A lookup now matches the tracker's canonical name and asks for the colour of the crop
without suggesting the requested answer. It caches that answer on the track, not on every camera frame.
In a separate apartment run, "What do you see?" got one reply, then "Go to the blue bicycle" took one drive,
no search and no bumps. The chosen box matched instance 4 (IoU 0.98); the endpoint was 0.36 m from its edge,
3.03 m from the other bike. This tests one viewpoint; it does not establish general colour recognition accuracy.

#### A thing first asked for from right beside it (`runs/goto-unseen`, 20 November)

In interactive session `220908-xKr5` the model drove `go_toward` to the blue bicycle, then `go_to("blue bicycle")`
said none was known or in view; the search it was told to call drove 3.5 m away, and of the nine calls after it,
two "blocked" answers were the harness's own. The probe (`runs/goto-unseen/probe.py`) repeats that from a fresh robot
with no model: the same drive, then the tool call alone.

- **Too few frames with the new word.** `detect` waited for two frames per camera, whatever words they were looked
  at with; the first could still run with the old ones. One front frame named the bike, one sighting is not a thing
  (two are), and `go_to` answered "not in view" with the body 0.19 m from the bike's edge. Each camera now looks
  three times with the word.
- **Too near to name.** From that close YOLOE named no bike in any of 24 frames (8 a camera, all with "bicycle"), yet
  the vision model said yes to the front picture with 95% ("is the blue bicycle in this picture": 100%; 15% for a
  picture without it). The search only asked about cameras with floor to drive to, which the bike itself took up
  ahead, so it went rear-left on 7% and found the bike from afar (2 of 2 runs got there, in 63 and 124 s). It now
  asks every camera, and when the likeliest has no floor steps back 1 m still facing that way: 4 of 4 runs did that
  once (front 93-95%, the others 3-25%), saw one blue bike, and parked 0.22-0.26 m from its edge in 46-54 s.
- **`go_to` searches for a name it does not know** instead of answering "use search": the model need not make that
  extra turn, nor know to.
- **Two "blocked" answers that were not.** go_toward accepted points the planner reached within 0.25 m, and go_to
  then planned to within 0.1 m: 276 of 14,992 go_toward points on the apt_0 map (1.8%) were refused; go_toward
  now accepts only what go_to will plan (0 refused). And a bike seen from about 4 m was mapped at (1.8, -2.3),
  0.6 m into the wall's margin, where no floor lay within reach; `go_to` now drives to the nearest floor it can
  reach when that is within a metre more (here 1.04 m from that centre, 0.51 m from the bike's later one). This
  last one is checked on the map and in the 2D simulator, not yet in a 3D run.
- **A step back that does not move must not repeat.** With no floor behind it either, a search asked the same
  question from the same spot 20 times ("red wall", front 84%); it now takes the likeliest way with floor instead,
  or ends.
- **One set of direction words.** The model used `look`'s camera words for `go_toward` ("rear-right") and the
  other way round (`look("front-right")`), each costing a turn; both tools now take both, each word mapped to a
  camera that sees that way. Unit-tested only.
- **A dead vision server fails plainly.** It died twice in this session with nothing in its log (a client
  dropping its connection, the first guess, did not kill it); every frame then failed quietly, `detect` said
  "not in view" and the depth node's scan stopped (no localisation). `detect` now raises when no picture was
  looked at in 6 s. Why the server died is not known.

With the model (`runs/goto-unseen-agent`, `-agent2`; one run each, so noisy): "Go to the blue bicycle." from the
dock passed 3 of 4 (one or several calls); the fourth reached the bike (0.19 m inside its footprint) but went
on calling tools until the 16-turn limit, after two approaches were cut off by "not there when the cameras last
looked" while the bike was there: tracks of things YOLOE names in few frames are given up too early, the same
message hit 6 of 23 tasks in `real2`, and some of those (false TVs) were right to go. Parked beside it first
("drive about 1.5 m front-right", then "go to the blue bicycle") passed 2 of 2 with one `go_to`, though the
shared detector still had "bike" from the earlier tasks (the probe above is the new-word case). Of the tasks
this touches, N1 and S3 passed (S3 in 3 s, the model answering without a call); N5 and S1 failed as they have
before (N5 passed 2 of the last 5 GPU runs): every "TV" known was a table or rug, and S1 kept choosing a "plant"
never seen again when faced, with 21 bumps turning in place by the furniture (8 and 5 in its last two runs).

#### Requests nobody tuned on (`runs/explore-1` to `-5`, 21 November)

Eight new requests in apt_0 (`runs/explore/new_tasks.py`: the beanbag, the umbrella, the shoes, the kitchen, facing
the sofa, the monitor, the middle of the rug, the stool), each checked against the layout's truth, run, read, fixed
and run again. What made the robot look dumb, in the order found:

- **Real things given up as gone.** "Go to the beanbag" made six `go_to` calls; five ended "object N was not there
  when the cameras last looked" while the model saw the beanbags in its pictures, and the task ran out of time 3.6 m
  from them with 26 bumps. Six detector misses in a row had made a track gone. The vision model now looks first at
  the part of the picture where the thing is mapped (`place_picture`): no makes it gone, yes keeps it. Reports had
  2 real things (a bike seen 110 times, a plant 53) and 2 false TVs dropped this way; sightings did not tell them
  apart (the false TVs were seen 60-103 times), the model's answer can (below). Afterwards no real thing was
  dropped; a false umbrella (a beanbag) and false shoes (a towel) were, on the model's no.
- **Arrivals refused.** Parked about 1 m from a real beanbag, `go_to` answered "target not confirmed" seven times:
  it wanted a detector sighting from the last instant, and YOLOE had not named the beanbag for 6-160 s. The
  vision model's yes on the thing's place now confirms an arrival too (`sees_now`). The same request then took 5
  turns and 39 s instead of 16 turns and 196 s, ending 0.86 m from a beanbag's edge: as near as the robot's map
  allows, the beanbags standing in floor it mapped as taken (the check wants 0.75 m).
- **A goal Nav2 could not reach.** A search stop free on the robot's map but not on Nav2's costmap kept Nav2 backing
  up, spinning and replanning along one 2 m stretch for 490 simulated seconds. A Nav2 goal that brings the robot no
  0.25 m nearer in 30 simulated seconds is now given up as blocked; the search then goes on elsewhere.
- **New things checked last.** With the "gone?" questions ahead of naming new things on the vision model's one
  queue, the bike YOLOE called an umbrella was driven to before it was checked; the model's yes to that crop is 0%
  (real beanbags 100%, a beanbag called an umbrella 3%). New things are now named first.
- **Step backs that cannot help.** The search stepped back on a 17% yes, and backed away from "kitchen" again and
  again (90-100% yes ahead, nothing YOLOE can name). It now steps back only on a yes of at least half, once a search.
- **"area" taken for a direction** (`search(area="rear-right")`, in two tasks): its description now says the
  search looks every way itself.
- **"Turn to face the sofa" drove off.** The search for a name not in view, added to `go_to` above, ran for every
  tool that takes a target; `face("sofa")` with the sofa 4.9 m off and unnamed drove away to look for it. Only
  the tools that go to the thing search now; `face` and `keep_facing` answer that it is not in view.

With all of it, "turn to face the sofa" took one call and 11 s, and "go to the stool" one call and 15 s, to one of
the bar-height seats by the bike that the vision model had called "stool-like chairs" and the layout calls chairs
(a check that names one kind of thing, not the robot, failed there). What these fixes do not reach is what the robot can
name and place at all, and three of the eight failed on it while the vision model kept answering 72-100% that the
thing lay ahead: things YOLOE does not name from 22 cm, a place that is not one thing, and a flat thing on the floor.
It never found the umbrella or the four shoes (it passed 1 m from the umbrella) and named 13 other things "shoes"
(towels, cloths, a chest of drawers, a rack, a chair, the bike, the TV), some of which the vision model's yes let
through; it never named "computer monitor" or "monitor" (that task passed only because time ran out with the robot
0.75 m from it); the open-plan flat is one room to the robot (named "office" by the vision model), so "go to the
kitchen" became a search for a kitchen-shaped thing; and a rug is dropped as floor by design (`locate`). Only the
detector turns a sighting into a place to drive to; the vision model, which sees these, cannot. Words left in the
shared detector from earlier tasks also fill the object list ("sofa": 9 tracks for 1 sofa). Bumps, 7 contacts and
42 ticks in one task, all came from turning in place by the coffee and desk tables, whose legs and frames the depth
network does not see (the hazard below).

#### When a navigation target disappears (interactive session `173110-YTOF`)

`go_to("tv 25")` kept driving to a fixed point even after the track disappeared. Nav2's logged start and goal
were 7.23 m apart. Replanning those points on the saved inflated map gave a 13.12 m route; 35 of 101 samples
on the straight segment were blocked. That explains a detour, not why the robot believed a TV was there.
The later lookup of ID 25 explicitly reported clear views without the object. The session did not keep its original
crop, so neither an initial false label nor a bad map position can be ruled out. Nav2 eventually reported failure.

In `guarded-final-IWne` a controlled verifier withdrawal after 0.15 m of an actual apartment drive cancelled the
accepted Nav2 goal, with no bump. The estimated pose moved another 0.10 m in the next 0.75 simulated seconds:
goal cancellation is not an instantaneous physical stop. In `view-goal-Q5jq` the right-hand chair's actual crop
and the rear-right image with both badges reached the model. Despite that, the model substituted a normal approach
for a request to go underneath. The bounded TV repeat also failed to find the real TV. These results validate
the evidence and cancellation paths, **not** safe under-furniture navigation or reliable recognition of screens.

#### Driving to things the vision model sees but YOLOE cannot (`vlm-locate`, 21 November)

When YOLOE returns no match for a requested thing, room area or floor surface (`RealRobot.detect`), the robot now
asks the vision model (`LFM3-VL-2B`) across its cameras, locates the target via yes/no crops (`locate_in_picture`),
ranges the crop (`locate_crop`), and registers a `visual=True` track that `go_to` and `sees_now` handle like any
other object — with no word lists, routers or per-object code.

- **768 px gate and overlapping yes/no crops (`robot/ros/perceive.py`).** Whole-image `see_probability` at 768 px
  (~0.35 s/frame) scored 98–100% on monitor, kitchen, rug, umbrella and shoes where 512 px scored 5–47%. To find
  where without numbered dots (which the 2B model picks last), `locate_in_picture` asks yes/no on overlapping
  thirds `(0–40%, 30–70%, 60–100%)`. Wide targets spanning contiguous positive thirds take the probability-weighted
  centre (cut close-up rug centre error from 0.78 m to 0.36 m); narrow targets split the best third into
  0.65-overlapping halves (non-overlapping halves sliced a centred umbrella at `u=0.50` from 96% to 4%/4%, while
  0.65 overlap gave 100%/99%), then an upper/lower split on a 0.30-span strip (`on_floor` only when
  `v_bot >= 0.5 and v_top < 0.35`, so standing objects like an umbrella at 89%/100% stay 3D). Offline benchmark on
  30 random `apt_0` frames (210 pairs): **0/145 (0.0%) false positives**; blank-wall hallucinations (71% whole)
  dropped to 1%/0% on crops.
- **Coarse-crop and flat-floor ranging (`robot/perception/locate.py`).** `locate_crop` samples the middle 50% of
  columns at the 25th depth percentile for standing/elevated targets (cutting `dock -> kitchen` edge error from
  1.50 m to 0.00 m) and records median elevation `z`. Floor-flat targets require ≥60% below-horizon floor pixels
  (`<= 0.06 m`) and weight floor pixels by `r^2` (exact flat-floor perspective Jacobian `dr ∝ r^2 dv`) over
  `[0.4, 3.5] m`: unweighted median fell at 0.72 m near the feet, linear `r` at 0.88 m, `r^2` to 3.0 m at 0.69 m
  (1.02 m after Nav2's 0.25 m tolerance), and `r^2` to 3.5 m plus the score-weighted `[2 cams]` centroid landed
  `big rug 1` dead centre on `rug 0`.
- **Mid-leg snapshots, visible-side approach and arrival confirmation (`robot/ros/real.py`).** During a search,
  `_trail_views` snapshots every 0.25 m (4 m window) catch targets visible only in narrow corridor gaps (the
  umbrella is visible over ~0.30 m along the hallway); mid-leg scans require `>= 0.80` to correct for testing up
  to 48 frames/stop (distant false hits scored 54–69%, real passes 97–100%). Standing visual tracks store
  `spotted_from` at the spotting camera's map coordinates so `_spotted_standpoint` and `_go` approach from the
  unobstructed side (stepping straight onto it with `drive_on_heading` if Nav2's 0.25 m tolerance stops 0.10–0.40 m
  short behind a wall corner). `sees_now` waits up to 1 s for a fresh post-`face()` frame, confirms elevated
  targets (`z > 0.45 m`, e.g. desk monitor invisible `< 1.15 m`, 76–89% at 1.5–1.7 m) via a closer approach
  snapshot, and confirms floor-flat targets (`radius == 0.0`) via the floor half when furniture sits on top of the
  surface (whole frame 47.3% with cabinet above, floor half `is_there=True`).
- **Frontier search, leaner harness and tracker fixes (`robot/ros/real.py`, `robot/agent/`, `robot/perception/tracker.py`,
  `robot/ros/robot.py`).** Wiring `RealRobot.walls = grid` and ranking `_unseen()` frontiers by A* path length
  capped to 3.2 m legs (`_frontier_step`) when no camera has `>= 50%` lead cut a full `apt_0` sweep from 14+
  living-room zigzags (406 s, 26 bumps) to 4–5 stops (~93 s to umbrella, 0 bumps). Default reasoning is now `low`
  (`--reasoning-budget 128`, median model turn 5.5–5.8 s); `review=False` default (auto-reviewing open `todo`
  steps) and `say` deduplication removed double replies; refreshing TF pose in `RosRobot.begin()` and prioritizing
  `"where you were before"` fixed `"come back"`; `MAX_OBJECTS=8`, `MAX_IMAGES=3`, lean stationary `look`,
  `search(..., toward=...)`, and `blocked_tools` (hiding any tool that hit `Not run:` until the robot moves) cut
  context bloat and stopped `look`/`look_back` repeat loops. `RosRobot._check_goal` now matches `Tracker.gate`
  (`max(0.6, r1 + r2)` — previously `2*r = 0.08–0.22 m` falsely aborted 4 consecutive `go_to` calls on small shoes
  refining depth by 0.16–0.26 m); `Tracker._merge` skips absent tracks and preserves `verified`/`sure`/`visual`;
  `RealRobot.obj` refreshes `_perceive_objects()` on worker-thread merge races; and `home3d.says_no` fixed `\bn't\b`
  which could never match contractions like `"don't"`.
- **What did not work (dropped).** Close-range floor refinement in `RealRobot._go` averaged bare wood behind the
  rug out to 3.5 m and drove *off* the rug in 2/2 runs (deleted, -40 lines). Rejecting `[2 cams]` centroids on
  occupied cells dropped dock `[2 cams]` because a cabinet sits on `rug 0`'s centre. Backing `_spotted_standpoint`
  out to `reach + 0.25` (0.90 m) stopped 1.13–1.15 m from E2's umbrella (tightened to `reach`). Euclidean-only
  frontier selection jumped 12 m blind through walls.

Measured 3D results in `apt_0` (repeated, `think=low`, 0.33x real time):

- **Close-up blue bicycle probe (`runs/goto-unseen/probe.py`, `vlm-probe`, 3 trials):** **3/3 PASS** in
  16.8–33.9 s with 0 search stops (gaps `0.35`, `-0.23`, `-0.00 m`; before: 46–54 s with a step-back search).
- **Unseen requests (`runs/explore/new_tasks.py`, repeated across `vlm-r3a`, `vlm-f1`, `vlm-finalA`, `vlm-finalB`):**
  - `E4` (kitchen): **4/4 PASS** (19.8–39.6 s, 2–4 turns, **0 bumps**, `-0.16` to `-0.59 m` inside counter edge;
    before: FAIL).
  - `E6` (computer monitor): **4/4 PASS** (13.9–15.5 s, 2 turns, **0 bumps**, `0.55–0.58 m`; before: timeout luck).
  - `E3` (shoes): **2/4 PASS** (`113.8 s / 0.32 m` and `177.4 s / 0.38 m / 0 bumps`; before: FAIL with 13 false
    YOLOE shoes). The 2 misses were YOLOE calling a hanging cloth `shoes` (1.35 m) and a pole occluding the rack
    at 3.4 m (2.54 m).
  - `E2` (umbrella): passed in `vlm-low3`, reached `0.84 m` (`vlm-fast3`, 0 bumps) and `1.13–1.15 m` in 3–5 turns
    / 0 bumps (`vlm-final1`, `vlm-f1`) once spotted on Stop 4; remaining variance is the ~0.30 m corridor visibility
    window and YOLOE occasionally hallucinating an umbrella on the chest of drawers/bike (rejected by `sees_now`).
  - `E7` (middle of big rug): **0/4 on `middle_of(0.9)` check, 3/3 on-rug in 22.8–31.1 s / 2 turns / 0 bumps**
    (`vlm-f1`, `vlm-finalA`, `vlm-finalB`), landing `-0.24` to `-0.28 m` *inside* `rug 0` at `0.97–0.98 m` from its
    geometric centre — flush against the 0.83 m-radius wooden cabinet covering `rug 0`'s dock-facing centre (raw
    un-inflated Habitat navmesh there only reaches `0.71 m`; before: `locate()` dropped rugs completely).
- **Non-regressions and false-positive checks (`vlm-reg`, `vlm-home`):** `E1` (beanbag, 21.0 s, 0 bumps), `E5`
  (face sofa, 8.2 s, 0 bumps), `N1` (sofa, 20.1 s, 0 bumps) and `S2` (dog, absent, 3.0 s, 0 bumps) all **PASS**.
  `S3` (elephant, absent) searched 7 stops with **0 false detections** (0–12% across all views, 0 bumps) and
  replied `"I don't see any elephant..."` (failed only due to the `\bn't\b` regex bug in `says_no`, now fixed).
  `R2` Turn 2 (`"Come back."`) succeeded in 1 tool call (`go_to('where you were before')`), ending 0.28 m from start
  (Turn 1 hit YOLOE's pre-existing TV flicker). `E8` (stool, 0 bumps) hit YOLOE's chair-as-stool without visual
  fallback; `S1` (plant, 0 bumps) genuinely spotted and drove to the foot of the stairs below the real upstairs
  potted plant (`010_candidate_33.jpg`).

#### Synchronous vetting, semantic place map and badge de-overlap (`vlm-locate`, 22 November)

Three general-purpose improvements built on `vlm-locate` (keeping `think="low"` at 128 tokens — LFM3-VL-2B's model card
explicitly advises short low-temperature tool routing backed by memory/retrieval rather than deep ungrounded reasoning):

- **Synchronous VLM vetting & bounded detector words (`robot/ros/real.py`, `robot/perception/detector.py`).**
  Previously `where()` returned brand-new unverified YOLOE tracks before the background `_verify_loop` saw them, and
  `detect()`'s 3 s polling loop timed out whenever unrelated tracks were queued. `RealRobot._matching()` now vets
  un-`sure` (or >=2x larger crop) non-person, non-visual YOLOE tracks synchronously via `is_there()` (deleting the 3 s
  polling loop in `detect()`), and vets synonym-matched labels (e.g. `"desk"` vs `"table"`) without destroying the
  underlying track. `Detector.asked` is capped at `MAX_ASKED = 3` most recent words even on high-slot servers so old
  task words stop leaking into new tasks.
- **Semantic place map built by the robot's own cameras (`robot/ros/explore.py`, `robot/ros/real.py`,
  `robot/sim/stack.py`, `robot/agent/`).** `Explorer` / `annotate_map_places` / `RealRobot._note_place()` caption up to
  `MAX_PLACES = 6` well-spaced (`PLACE_GAP = 2.8 m`) viewpoints with `describe_place()` (~50–75 ms per caption on
  LFM3-VL-2B, saved beside `maps/<scene>.yaml` in `maps/<scene>.places.json`, zero truth leakage). `Agent._places_text`
  lists `"where you were before"` first, `"start"` second, then mapped places nearest-first; `Tools._resolve` fuzzily
  matches mapped areas by word overlap for `search(near=...)` and out-of-view `go_to` before blind sweeping.
  `RealRobot.search()` preserves per-target coverage across calls within one request (cleared in `begin()`),
  `Tools.search()` reports unvisited mapped areas (`> 2.2 m` from trail), and `Agent.do()` blocks identical repeat
  `search()` calls after moving.
- **De-overlapped ID badges & co-located clustering (`robot/perception/overlay.py`, `robot/agent/tools.py`).**
  `_place_badge()` nudges colliding badge rectangles apart vertically/horizontally so adjacent IDs never fuse into
  `"1920 SHOES"` (verified live on `E3`: model read `"17 SHOES", "24 SHOES", "20 SHOES"` cleanly). Same-label matches
  within `CLUSTER_GAP = 0.7 m` auto-pick the nearest instead of raising an ambiguity error.
- **ROS 2 action-client discovery fix (`robot/ros/robot.py`).** `RosRobot.__init__` previously built `ActionClient`s
  before `_wait_for_nav2` finished activating `behavior_server`, causing a FastDDS SHM discovery miss on
  `drive_on_heading` right after `E2` reached `0.84 m` from the umbrella (`vlm2-b`). Creating `self.actions` after
  `_wait_for_nav2` and destroying clients before recreating and on `shutdown()` eliminated both the `drive_on_heading`
  error and the `sprocket_actions` shutdown warnings (`vlm2-c`).
- **What did not work (fixed from measurements).** A 3-camera `320x180` panoramic strip in `describe_place` collapsed
  open-plan views into `"living room: couch, coffee table"` because the rear cameras looked back at the sofa; switching
  to the single `768 px` `front_camera` (simpler, sharper) separated living room, kitchen, hallway and staircase on
  `rand30.npz` and produced 6 distinct nodes on `maps/apt_0.places.json`. Sampling at `0.45 m` clearance placed a tour
  node behind the west coffee table (`-6.25, -1.03`) where leaving snagged on the thin legs; raising open-floor
  clearance to `0.55 m` plus backup-clear recovery completed all 6/6 tour waypoints cleanly. Clearing `self.coverage`
  and `seen_calls` on every `search()` caused `vlm2-a`'s `E2` to re-sweep the living room 3 times; preserving per-task
  coverage and keeping `search` keys in `seen_calls` stopped the loop.

Measured 3D results after these fixes (`vlm2-b`, `vlm2-c`, `vlm2-home`, `vlm2-n1`, `vlm2-e15`, `think=low`, 0.33x):

- `E2` (umbrella): **2/2 reached real umbrella with 0 bumps** (`vlm2-c` **PASS** in `128.1 s` / 3 turns / `0.79 m`,
  35% faster than Phase 1; `vlm2-b` reached `0.84 m` in 2 turns / 0 bumps before the `drive_on_heading` fix).
- `E4` (kitchen): **2/2 PASS** (`22.7 s` / 2 turns / 0 bumps / `-0.27 m`; `93.1 s` / 4 turns / 1 bump / `0.84 m`).
- `E6` (computer monitor): **2/2 PASS** (`15.2 s` / 2 turns / `0.55 m`; `15.6 s` / 2 turns / `0.58 m`).
- `E3` (shoes): **1/2 PASS** (`vlm2-b` **PASS** in `163.4 s` / `0.28 m`, cleanly reading de-overlapped badges;
  `vlm2-c` timed out after a false visual `[passed]` shadow on a rug and a reviving false YOLOE track in the west
  corner).
- `E7` (big rug): **2/2 drove onto `rug 0` with 0 bumps** (`-0.22 m` in `32.1 s` / 2 turns; `-0.26 m` on Turn 2 using
  `search(what='big rug', near='living room')`; same cabinet-edge limit vs `0.90 m` check).
- Non-regression & FP (`vlm2-home`, `vlm2-n1`, `vlm2-e15`, **0 bumps across all runs**): `S2` (dog, **PASS**, `5.0 s`,
  0 FP), `S3` (elephant, **PASS**, `3.2 s`, 0 FP), `R2` (TV + `"Come back."`, **PASS**, `99.0 s`, 1-turn return to
  `0.1 m` of start), `N1` (sofa, **PASS**, `20.5 s`, 2 turns, `0.66 m`; one earlier 1-off raw-JSON 2B reply),
  `E1` (beanbag, **PASS**, `49.1 s`, 2 turns, `0.64 m`). `E5` (`Turn to face the sofa.`) had 0 bumps in `18.9 s` but
  the 2B model chose `go_to('sofa')` instead of `face('sofa')`.

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
