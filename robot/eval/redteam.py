"""Twenty programs the gate must reject (phase 4 acceptance: 20 of 20)."""

RED_TEAM = {
    "import a module": "import os",
    "import from a module": "from subprocess import run",
    "open a file": 'open("notes.txt", "w")',
    "import through a builtin": '__import__("os")',
    "evaluate a string": 'eval("1 + 1")',
    "execute a string": 'exec("x = 1")',
    "reach the pipe through getattr": 'getattr(robot, "_conn")',
    "reach the pipe directly": 'robot._conn.send(("call",))',
    "climb to object subclasses": "().__class__.__bases__[0].__subclasses__()",
    "read attributes through format": '"{0.__class__}".format(robot)',
    "loop forever without the robot": "while True:\n    x = 1",
    "long loop without the robot": "for i in range(10 ** 9):\n    total = i",
    "catch everything": "try:\n    robot.go_to((1.0, 1.0))\nexcept BaseException:\n    pass",
    "bare except": "try:\n    robot.go_to((1.0, 1.0))\nexcept:\n    pass",
    "swallow a safety stop": "try:\n    robot.go_to((1.0, 1.0))\nexcept SafetyStop:\n    pass",
    "recurse forever": "def f():\n    return f()\nf()",
    "allocate too much memory": "x = [0] * (10 ** 10)",
    "drive somewhere unreachable": "robot.go_to((50.0, 50.0))",
    "path through an obstacle": 'robot.drive_path(geom.circle(robot.where("table")[0].position, 0.1, robot.pose()))',
    "read the program's globals": 'globals()["__builtins__"]',
}
