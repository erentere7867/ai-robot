"""Allow running the harness via `python -m robot`."""

from robot.main import main

if __name__ == "__main__":
    raise SystemExit(main())
