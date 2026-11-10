"""Plausible but wrong programs: each must fail its task's check.

They are the mistakes the untrained planner actually made, or the nearest wrong reading of the instruction. An idle
robot failing every check is not enough: a check must also reject a program that does something close but wrong, or
fine-tuning against the checks (GRPO especially) learns to exploit it.
"""

WRONG = {
    "R1": ['robot.move(metres=1.0, direction="forward")\nrobot.spin(turns=2)\n',
           'robot.move(metres=1.0, direction="backward")\nrobot.spin(turns=1)\n'],
    "R2": ["robot.drive(metres_per_second=0.3, seconds=1.5)\n", "robot.move(metres=1.0)\n"],
    "R3": ['robot.turn(degrees=90, direction="right")\n', 'robot.turn(degrees=45, direction="left")\n'],
    "R4": ['robot.turn(degrees=90, direction="left")\n'],
    "R6": ['robot.move(metres=0.5)\nrobot.turn(degrees=45, direction="left")\n'],
    "R7": ['robot.spin(turns=1, speed="fast")\n'],
    "O1": ['robot.go_to(robot.where("sofa"))\n'],
    "O3": ['robot.orbit(robot.where("table"), laps=2, direction="left")\n',
           'robot.orbit(robot.where("table"), laps=1, direction="right")\n'],
    "O5": ["robot.go_to(robot.where(\"box\"))\n"],
    "O6": ['robot.go_to(robot.where("ball"))\n'],
    "O7": ['robot.go_to(robot.where("plant"))\nrobot.go_to(robot.where("lamp"))\n',
           'robot.go_to(robot.where("lamp"))\nrobot.go_to(robot.where("plant"))\nrobot.go_to(robot.user(), distance=1.0)\n'],
    "S1": ['robot.say("No, I did not find a backpack." if not robot.where("backpack") else "Yes.")\n'],
    "C5": ['robot.go_to(min(robot.where("box"), key=lambda b: b.distance))\n'],
    "F1": ["robot.face(robot.user())\nrobot.wait(60)\n"],
    "F2": ['robot.say("You are in the living room.")\n', "robot.face(robot.user())\n"],
    "M4": ['robot.go_to(robot.where("box"))\n'],
    "A1": ['robot.go_to(robot.where("box"))\n'],
    "A2": ['robot.move(metres=1.0)\n'],
    "A4": ['robot.go_to("kitchen")\n'],
}
