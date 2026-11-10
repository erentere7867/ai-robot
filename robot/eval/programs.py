"""Hand-written programs for suite instructions: proof that the API, the skills, the simulators, the gate and the
runner can do the tasks before any model writes a program. They are test fixtures, never shown to the planner."""

PROGRAMS = {
    "R1": """\
robot.move(metres=1.0, direction="backward")
robot.spin(turns=2)
""",
    "R3": """\
robot.turn(degrees=90, direction="left")
""",
    "R7": """\
robot.spin(seconds=5, speed="slow")
""",
    "O1": """\
chairs = robot.where("chair")
if chairs:
    robot.go_to(chairs[0])
    robot.say("I'm at the chair.")
else:
    robot.say("I can't see a chair.")
""",
    "O3": """\
robot.orbit(robot.where("table"), laps=2, direction="right")
""",
    "O4": """\
robot.go_to(geom.between(robot.where("sofa"), robot.where("table")))
""",
    "O5": """\
robot.go_to(robot.spot(robot.where("box"), side="behind"))
""",
    "O6": """\
robot.save_place("start")
robot.go_to(robot.where("ball"))
robot.go_to("start")
""",
    "S1": """\
bag = robot.search("backpack", area="everywhere")
robot.say("Yes, there is a backpack." if bag else "No, I did not find a backpack.")
""",
    "S6": """\
mug = robot.search("mug", area="kitchen")
if mug:
    robot.go_to(mug, distance=0.3)
    robot.say("I'm next to the mug.")
else:
    robot.say("I couldn't find a mug in the kitchen.")
""",
    "C1": """\
doors = robot.where("door")
if doors and robot.check("Is the door open?", doors[0]):
    robot.go_to(robot.spot(doors[0], side="behind", distance=1.0))
else:
    robot.say("The door is closed.")
""",
    "C5": """\
boxes = [b for b in robot.where("box") if b.color != "blue"]
if boxes:
    robot.go_to(min(boxes, key=lambda b: b.distance))
""",
    "F1": """\
robot.follow(robot.user())
""",
    "F2": """\
me = robot.user()
robot.go_to(me, distance=1.0)
robot.face(me)
""",
    "M4": """\
balls = robot.where("red ball")
if balls:
    robot.go_to(balls[0].position, distance=0.3)
    robot.say("This is where I last saw the red ball.")
else:
    robot.say("I don't remember seeing a red ball.")
""",
    "A1": """\
boxes = robot.where("box")
if len(boxes) > 1:
    answer = (robot.ask("I can see several boxes. Which one do you mean?") or "").lower()
    boxes = [b for b in boxes if b.color and b.color in answer] or boxes
robot.go_to(boxes[0])
""",
}
