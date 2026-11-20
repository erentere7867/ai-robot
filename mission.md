cl# Mission: finish Sprocket and make it as capable as it can be

You are taking over Sprocket's software for a long stretch of work: weeks, not hours. Sprocket is a small indoor
robot. It has three cameras, an on-board vision-language model that decides what to do through tool calls, YOLOE and
depth for fast perception, and Nav2 for driving. Everything is in this repository. It runs in a 3D apartment
(Habitat, ReplicaCAD) on its real software, with only the world, the body and the raw sensors simulated.

You have two jobs, and you switch between them all the time:

1. **Engineer.** Finish the software stack and make the robot as capable as the hardware allows. Use whatever design
   works best; section 2 says how free you are.
2. **Observer.** Watch the robot work in the apartment the way a demanding owner would, and judge it. The question
   is not whether a test went green. It is whether a person watching would call the robot competent and smart.

"Done" never means "the tests pass." You are done with a piece when the robot handles requests nobody wrote code
for, in layouts nobody tuned on, reliably and quickly enough to run on the real board, and the numbers show it.

This brief is `docs/mission.md`. If it is not committed yet, commit it in your first PR and link it from
`AGENTS.md`'s "Start here". Every later session re-reads it.

## 0. Before anything

Read these, in order:

1. `AGENTS.md`. Its rules bind every change you make.
2. `README.md`. It has a table of where each behaviour lives in the code.
3. `docs/usage.md` (this PC), then `docs/testing.md`.
4. `docs/architecture.md`. Section 6 is the agent as it is now. Sections 3 and 5 are the target design and only
   partly built.
5. `docs/simulation_3d.md`. It records what has been measured, including what failed and why. Never retry an idea
   it says failed without a new reason.
6. `docs/ideas/sprocket-harness.md`.

## 1. The machine: this PC

You test on the owner's Windows 11 PC, where Sprocket is developed. It has an AMD RX 6600 GPU and WSL 2 running
Ubuntu 26.04 with 6 cores. The pieces run in two places and talk over `127.0.0.1`:

| Piece | Where | Port | Start and stop | Log |
| :--- | :--- | :--- | :--- | :--- |
| Language model (LFM3-VL-2B on llama.cpp's Vulkan build, `C:\llama-vulkan\llama-server.exe`) | Windows, GPU | 8081 | `powershell -ExecutionPolicy Bypass -File tools\brain.ps1`, add `-Stop` to stop | `%TEMP%\sprocket-llama.log` |
| Vision server (YOLOE-26l and depth, ONNX Runtime with DirectML, venv `%USERPROFILE%\sprocket-win`) | Windows, GPU | 8090 | `tools\win\vision.ps1`, add `-Stop` to stop | `%TEMP%\sprocket-vision.log` |
| The robot (ROS 2 lyrical, Nav2, the perception nodes, the agent) and the apartment (Habitat-Sim) | WSL: `~/sprocket-venv`, `/opt/ros/lyrical`, `~/micromamba/envs/habitat`, `~/habitat-data` | live page 8765 | `python -m robot.eval.home3d ...`, then `bash tools/linux/stop.sh` | `runs/home_<scene>/`; Nav2's in `~/.ros/log/` |

Everything is installed and was checked on 19 November.

- **Check it first.** Start the two Windows servers, then in WSL run
  `. tools/linux/env.sh && python -m robot.eval.doctor --3d`. It should end with `ready`. If something has broken,
  fix it in the scripts, not by hand.
- **Which doc for which machine.** `docs/usage.md` is the reference for this PC. `docs/testing.md`'s route for a plain
  Linux machine should keep working, but this PC is where you test.
- **Speeds here.** The model reads a picture in about 0.15 s. YOLOE-26l takes 75 ms a frame and depth 82 ms, both on
  the GPU. One 3D task takes a few minutes at `--rtf 0.33` to `0.5`. A full pass of the tasks on the two tuning
  layouts takes hours, so run repeats overnight.

### Working here without losing runs

Each of these has cost a run before.

- **Run Python in WSL;** Windows' Python has no numpy.
- **Put WSL commands in a script file** and run it with `wsl -- bash /mnt/c/.../script.sh`. An inline
  `wsl -- bash -lc '...'` loses shell variables and quotes, and the repository's path has a space in it.
- **Source `tools/linux/env.sh` before anything that uses ROS.** Over UDP, ROS here finds its peers but delivers no
  data; `env.sh` switches it to shared memory. It also turns on GPU rendering for Habitat (`GALLIUM_DRIVER=d3d12`).
- **The GPU from WSL is for rendering only.** ROCm does not support the RX 6600 in WSL, and Vulkan in WSL runs on the
  CPU. Models that need the GPU run as Windows programs, as the two servers do.
- **Restarting your own session kills the processes it started.**
  - Start the model detached, from PowerShell:
    ```powershell
    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "<repo>\tools\brain.ps1"'}
    ```
  - Start long runs in WSL with `setsid nohup bash script.sh > log 2>&1 &`.
  - End the launcher with `sleep 3; pgrep -af <pattern>`. A detached launch dies if the `wsl` call that started it
    returns at once, and `pgrep` shows that it started.
  - When you wait for a run, make the loop also stop when the run's process is gone.
- **Stopping processes:**
  - rclpy processes ignore SIGTERM, so stop the stack with `tools/linux/stop.sh`.
  - A `pkill -f` whose pattern appears in its own command line kills itself. Write the pattern as
    `"[r]obot.eval..."`.
- **The model's context:** each request gets 16,384 tokens (`-c 32768 -np 2`). An overflow shows in the model's log
  as "exceeds the available context size".
- **One GPU, six cores.** One GPU serves the model, the vision server and Habitat's rendering, and WSL has 6 cores.
  Run one 3D evaluation at a time, and nothing heavy beside a run you are measuring: a YOLOE export during a run
  changed its results.
- **Pages:** from Windows, open them at `http://127.0.0.1:<port>`. `localhost` can time out.
- **The owner may be using this PC.**
  - Leave their programs and the system settings alone.
  - Keep your files inside the workspace.
  - Keep `runs/` from growing without limit: keep the baselines and the runs your PRs cite.

## 2. You may change anything

Nothing in the current design is fixed. If a different design is better, build it. That includes:

- the agent loop and its prompt;
- the tools;
- perception: YOLOE, depth, the synthetic scan, the tracker;
- navigation: Nav2's settings, or something other than Nav2;
- memory;
- how the model is served and called;
- the processes and how they talk to each other;
- the evaluation harness itself.

Big architectural changes are welcome when they buy performance, reliability, efficiency or accuracy that small ones
cannot.

**What does not change:**

- **General-purpose.** The robot must work in any home, for requests nobody wrote code for, and adapt to what it
  finds. No hard-coding for a scenario, task, word, room or layout, anywhere, in an old design or a new one.
- **The board.** Everything runs on the Arduino VENTUNO Q: an Adreno GPU, a 40 TOPS NPU, 8 Kryo cores and 16 GB.
  Sensors and chassis cost under $150, and there are no cloud models.
- **The simulator** simulates only the world, the body and the raw sensors (cameras, IMU, wheels, ToF,
  microphones). What the robot knows comes from its own software.
- **Evidence decides.**

**How to make a big change:**

1. **Write a short design note first,** in `docs/` or the PR description:
   - what is wrong with the current way, with numbers from the ledger;
   - the new design;
   - what it should improve, and by how much;
   - how you will measure it;
   - what it deletes.
2. **Build it on a branch,** in steps that each run.
3. **Measure it against `main`** on the same tasks, layouts and repeats.
4. **Merge it when it wins.** Remove the old way in the same change or right after: never two systems doing one job.
5. **Update `docs/architecture.md`** so it describes what is.

**Work to a real engineering standard.** Do harness, robotics and AI engineering the way experienced engineers in
those fields do, using what the fields already know. When a question has a known answer, read the sources: papers,
and the docs of the libraries you use. The examples below show the kind of work meant. They are not a list to
follow.

- **AI and the harness:**
  - development driven by evaluation;
  - tools and prompts designed and measured on replays;
  - deciding what the model sees each turn, and what it costs;
  - reusing the server's cache for the part of each request that does not change;
  - structured output where the model must pick from options;
  - quick judgements by the model where a skill needs one, but never inside a control loop.
- **Robotics:**
  - localisation that knows how uncertain it is;
  - maps that update when the world changes;
  - planning and recovery that report failure instead of hiding it;
  - what the three cameras see, fused over time rather than trusted from one frame;
  - watchdogs and health checks across the processes.
- **Systems:**
  - find out where the time goes before optimising;
  - per-frame work on the GPU or NPU, with fixed input sizes;
  - no work spent on frames nobody uses.

## 3. How you work: one cycle per change

1. **Measure.** Run the tasks on current `main` with a clear tag (for example `base-1121-<commit>`), at least 3
   repeats.
2. **Watch.** Read the reports as described in section 4, and write a verdict for each task.
3. **Count.** Put the failures into the ledger by cause: harness, perception, body, check or model. Pick the
   biggest *general* cause, not the most recent one or the most interesting one.
4. **Reproduce cheaply.** Use the smallest test bench that shows the problem (section 5).
5. **Change.** Make the change that removes the cause in general, whether small or architectural (section 2). Delete
   what it replaces.
6. **Check:**
   - ruff and pytest, with a unit test for any logic;
   - the bench;
   - repeated 3D runs of the affected tasks and of the tasks the change could break, on the tuning layouts;
   - the held-out layouts, at milestones.
7. **Ship.** Open a PR with the cause, the before-and-after numbers and the run tags.
8. **Write it down** in `docs/progress.md` and, for findings about the robot, in `docs/simulation_3d.md`.

## 4. Watching the robot

You are the intelligence watching over the robot. For every failed task, and a sample of passed ones in every batch:

- **Open the report:** `runs/<tag>/<scene>/<task>/report.html` and the pictures in that folder.
  - `map.png` shows the real objects against what the robot believed, and where the body really went against where
    the robot thought it was.
  - Each model turn shows its reasoning, its calls and their results, and the exact pictures it was given.
  - The Problems section lists what went wrong outside the turns.
- **Read the logs** when the report is not enough:
  - `runs/home_<scene>/` (the robot's processes);
  - `~/.ros/log/` (Nav2);
  - `%TEMP%\sprocket-llama.log` (the model);
  - `%TEMP%\sprocket-vision.log` (YOLOE and depth).
- **Watch some runs live** with `--live 8765`, at http://127.0.0.1:8765, in a browser. Watching shows things a report
  flattens: hesitation, spinning, a camera that freezes.
- **Look at the pictures the model got, and answer its question yourself.** If you cannot tell either, the fault is
  in the task, the check or the camera, not the model. A fridge drawn as a flat black panel was one such case.
- **Write a verdict for each task:**
  - what was asked;
  - what the robot did;
  - what a sensible robot would have done;
  - the first wrong step;
  - the cause;
  - the evidence: turn number, picture, log line.
- **Judge passes too:**
  - lucky passes: it ended in the right place by accident, or said "yes" without having looked;
  - wasted motion: spinning, going back and forth, driving past the target;
  - false statements;
  - needless questions;
  - slowness: model turns, tokens, simulated time.

  A pass that a person would find dumb is still a failure of intelligence. Put it in the ledger.
- **Count patterns, not anecdotes.** The ledger in `docs/progress.md` lists the top causes, with counts and one
  example report each.
- **Use helpers if your harness can start them (subagents).** Give each one a batch of reports and this section as
  its instructions, then check a sample of their verdicts yourself.

## 5. Test benches: reproduce and measure cheaply

- **Perception:**
  - `python -m robot.eval.views render --scene apt_0 --count 40` renders views from random spots, with the truth of
    what is in each;
  - then `views yoloe ...` or `views model ...` scores either one per word: found and false.
- **Model decisions: build a replay bench in your first week.**
  - Save the exact request of every model turn with the run: messages, tools, pictures.
  - Add a command that re-sends chosen turns, N samples each, with a changed prompt, tool description or result
    wording, and reports how the choices change.
  - Keep a fixed set of hard turns from the ledger as a regression set.
  - This turns "did that prompt change help?" from hours of 3D runs into minutes. It is the main lever on the 2B
    model's tool choice.
- **The 2D suite** (`python -m robot.eval.suite`): fast checks of tool choice in drawn rooms.
- **One 3D task** (`--only <id> --live 8765`): for the body and how the pieces fit together.
- **Unit tests** for logic, with a fake robot or a scripted model, like the existing ones.

Use the cheapest bench that reproduces a problem, then confirm in 3D.

## 6. Measuring honestly

**What to record for each task run.** Add whatever is missing to the run's jsonl row and report, keeping the code
small:

- whether it passed;
- simulated seconds;
- path length against the shortest reachable path (success weighted by path length, SPL, as embodied-AI
  benchmarks use);
- bumps;
- model turns, prompt tokens and generated tokens;
- wall seconds per model turn;
- Nav2 failures;
- false beliefs: things believed with no real one near;
- false claims: saying something is there, or done, when it is not;
- repeated calls.

**One run is noise.** Between two runs of the same 23 tasks, 6 changed outcome. So:

- Run at least 3 repeats before calling a change better or worse.
- Report counts (7/9), not only percentages.
- Ship a change when it is better and not worse on any category beyond the noise.
- Write a command that compares two run tags, if none exists: pass counts and metrics per task, and which tasks
  flipped.

**Tuning against held-out:**

- Tune on `apt_0` and `v3_sc0_staging_00` (`TUNE` in `robot/eval/home3d.py`).
- Hold out other layouts, starting with `apt_4` and `v3_sc3_staging_07` and adding more. Run them only at milestones.
  Look at their numbers, not their reports, and never change code because of one held-out failure.
- If held-out lags tuning, you are overfitting. Find what is specific.
- ReplicaCAD's 90 layouts are one apartment with the furniture moved. Truly different homes need another dataset,
  such as HSSD or HM3D, whose licences the owner has to accept. Ask the owner when you reach that point.
- A new layout needs its map first: the robot explores it (`robot/sim/stack.py`) and saves it in `maps/`. Commit the
  maps of the layouts you use, as `maps/apt_0.*` is.

**The checks must stay honest:**

- Never change a check, a threshold or a task so that the robot passes.
- Fix a check only when it disagrees with the truth. Say so in the PR, and score earlier runs with the fixed check.
- Write a new task's check before you look at how the robot does on it.

## 7. What to build: milestones

This order is a default. Reorder it by what the ledger says. Each milestone ends with its exit criterion measured as
in section 6.

1. **Instruments and a baseline (first week).**
   - The doctor says `ready`.
   - The replay bench, the scoreboard metrics and the comparison command exist.
   - Baseline: 3 repeats on the tuning layouts and the held-out ones.
   - `docs/progress.md` is started.
2. **More and better tasks.** The 27 tasks in `robot/eval/home3d.py` are too few to steer by. Grow them to about 100
   that real people would say at home, across these kinds:
   - motion;
   - going to a thing;
   - places that are not one thing ("between the sofa and the table");
   - questions about what it sees, counting and comparing;
   - finding something not in view;
   - several steps, and conditions;
   - memory across requests;
   - refusing what it cannot do;
   - asking when a request is ambiguous;
   - "stop" and interruptions mid-task.

   Checks read only the truth, and work in any layout. Freeze a held-out subset of the tasks too.
3. **Reliable basics.** Targets:
   - motion: nearly 100%;
   - going to a thing in view, or one it knows: at least 90%;
   - an honest "no" or "I can't": at least 95%;
   - no false claims.
4. **Perception.** Three known problems, each measured with `views` before you change anything:
   - **False beliefs.** YOLOE's false boxes get past the model's yes/no check on the crop: six "sofas" were known in
     a flat with one.
   - **Furniture YOLOE cannot see from 22 cm.** It finds tables and counters 0–8% of the time. The model finds them
     90% of the time or more. Give the robot a general way to use that.
   - **The synthetic scan.** 64% of obstacles within 1 m are ranged within 15%.
5. **Decisions.** The 2B model picks the wrong tool more often as tools are added; there are 21 now. Try fewer,
   clearer tools, results and errors that guide it better, and a shorter prompt. Measure on replay and the 2D suite,
   then confirm in 3D.
6. **Search and exploration.** Find what is not in view, using rooms and where things usually are, without visiting
   the same place twice. Measure SPL and time.
7. **Body.**
   - Nav2 failures ("status 6").
   - Localisation about 0.5 m off far from the dock.
   - Contact with things no sensor saw.

   Build the behaviours in architecture section 3 only where a measurement shows they help.
8. **Memory.**
   - Replace the hand-written household lists in `robot/mind/memory.py` (`CANONICAL_SYNONYMS`, the anchor labels and
     the hashed-word "embeddings") with learned embeddings already in the stack, such as MobileCLIP's text encoder.
     Those lists break the first rule.
   - Remember across tasks and sessions.
   - Answer "where are my keys?" from what the robot saw, not only from what it was told.
9. **Voice and interaction, end to end.** Speech in (Whisper), then the agent, then speech out (Piper). "Stop" and
   interruptions mid-task. The wake-word turn and the voice direction, which are built but not wired in. Test it in
   the simulator by speaking requests with Piper into the Whisper path.
10. **People:** "follow me", "come here", "what am I pointing at". The apartment has no people. Find a way to put
    moving people in the simulator, such as Habitat's humanoids or a moving textured model, and test these honestly.
    If that is not feasible, report what you tried.
11. **The board's budget.**
    - Track tokens per turn (about 3,500 today, with three pictures), turns per task, and what that means in seconds
      on the board.
    - Run some tasks at `--rtf 1` to see what a world that does not wait does to the robot.
    - The model is provisional. Measuring other small vision-language models that could run on the board, through the
      same harness and checks, is in scope. Switching is the owner's decision, so bring numbers.
12. **Long runs.** A "day at home" of an hour or more: a stream of requests across rooms, with memory between them.
    Watch for crashes, drift, leaks and answers getting worse.
13. **Learning, later and only with the owner's agreement.** Fine-tune the model on recorded episodes from training
    layouts (`robot/train/scenes.py` makes random rooms), and judge it on layouts and tasks it never trained on.

**Keep the design honest.** Whatever in architecture sections 3 and 5 does not earn its place in measurement, remove
from the design and say why. `docs/architecture.md` describes what is.

## 8. Engineering, beyond AGENTS.md

- **One change per branch and PR.** The description gives the cause, the numbers before and after with run tags, and
  what you did not test.
- **Merging.** CI (ruff and pytest) must pass. You may merge your own PR when CI passes and the 3D evidence holds.
  Never force-push or rewrite `main`. Never commit models or run output.
- **Code shrinks where it can,** even when a redesign adds something new. Report the line count of `robot/` at each
  milestone. Delete what a change replaces. No second helper doing what another already does, and no options nobody
  uses.
- **Hot paths** (perception per frame, control loops): measure before and after.
- **Other agents open PRs here too.** At the start of each session, review the open ones against `AGENTS.md` and run
  them. Merge the sound ones; comment on the others saying why.

## 9. Your memory: docs/progress.md

Keep it short: replace what is out of date rather than appending forever. It holds:

- **Scoreboard.** The latest numbers per kind of task, tuning against held-out, with the date, the commit, the run
  tags and the number of repeats.
- **The milestone in progress:** its exit criterion and what is left.
- **Failure ledger:** the top causes, with counts and an example report each.
- **The next three things to do,** and why.
- **Experiment log:** one line per experiment, saying what it was, the result, whether it was kept or reverted, and
  the PR.

Read it first in every session, and update it before you stop. Your context will be reset many times, and anything
only in your head is lost.

## 10. Never

- Code for one task, word, room or layout. Lists of object or room names. Regexes or keyword routers over the user's
  words or the model's output. Rewriting what the model chose.
- Change a check, a threshold or a task so that the robot passes.
- Add fallbacks that hide a failure. A missing server fails plainly.
- Claim an improvement from one run.
- Retry what `docs/simulation_3d.md` says failed (numbered dots, a fixed word list, the fridge task) without a new
  reason.
- Let the simulator's truth reach the robot.
- Stop at "unit tests pass". The 3D apartment is the judge.
- Grow the code or the docs with filler.

## 11. Reporting to the owner

The owner is new to robotics and wants blunt, honest reports in plain words, not reassurance.

**At each milestone, or every week, write a short report:**

- the scoreboard then and now;
- what got better, and why;
- what did not, and why;
- what you would do next;
- a link or picture of one good run and one bad run.

**Decisions only the owner can make:** hardware, the choice of model, dataset licences and fine-tuning. Ask about
those. Decide everything else yourself, with evidence, and keep going. That includes big changes to the design.

**Start:** check the machine with the doctor, read the docs, measure the baseline, then milestone 1.
