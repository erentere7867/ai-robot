"""Sprocket's body as a Fusion model, generated from the dimensions below (all in millimetres).

Axes: X forward, Y left, Z up, origin on the ground under the centre of the wheelbase.
Run inside Fusion (Scripts, or the Fusion MCP `script` call) with `build()`; it rebuilds the model it made before.
"""

import math

import adsk.core
import adsk.fusion

MM = 0.1  # Fusion's API works in centimetres

# ---- drivetrain: 4 direct-drive hub motors (Waveshare DDSM115 class, 100 mm tyre) ----
WHEEL_R, TYRE_W, HUB_R = 50.0, 40.0, 40.0
WHEELBASE, TRACK = 260.0, 300.0
AX, AY, AZ = WHEELBASE / 2, TRACK / 2, WHEEL_R

# ---- suspension: independent trailing arm per wheel with a 90 mm coil-over ----
PIVOT_X, PIVOT_Z = 50.0, 80.0
ARM_Y0, ARM_Y1 = 93.0, 105.0  # the arm plate, measured out from the centreline
SHOCK_Y = 102.0
SHOCK_LO, SHOCK_HI = (118.0, 78.0), (78.0, 160.0)  # (x, z) of the lower and upper eyes

# ---- chassis ----
TUB_X, TUB_Y, TUB_Z0, TUB_Z1, TUB_WALL, TUB_CHAMFER = 112.0, 85.0, 45.0, 118.0, 3.0, 14.0
DECK_X, DECK_Y, DECK_Z0, DECK_Z1, DECK_R = 155.0, 113.0, 118.0, 122.0, 45.0
SHELL_X, SHELL_Y, SHELL_Z0, SHELL_Z1, SHELL_RV, SHELL_RT, SHELL_WALL = 160.0, 118.0, 116.0, 192.0, 50.0, 16.0, 3.0

# ---- head ----
NECK_X, HEAD_PIVOT_Z, HEAD_TILT = 80.0, 232.0, 10.0  # the head looks 10 degrees up
HEAD_X0, HEAD_X1, HEAD_Y, HEAD_H, HEAD_RV, HEAD_R, HEAD_WALL = -40.0, 58.0, 80.0, 104.0, 22.0, 16.0, 2.5
REAR_CAMERA_YAW = 120.0
LIDAR_TOP_OF_PEDESTAL = 343.0

app = design = root = tmp = lib = None
_looks = {}


# ------------------------------------------------------------------ geometry helpers (temporary bodies, mm)
def P(x, y, z):
    return adsk.core.Point3D.create(x * MM, y * MM, z * MM)


def V(x, y, z):
    return adsk.core.Vector3D.create(x, y, z)


def add3(a, b, k=1.0):
    return (a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2])


def unit(a):
    n = math.sqrt(sum(c * c for c in a))
    return (a[0] / n, a[1] / n, a[2] / n)


def box(xa, xb, ya, yb, za, zb):
    x0, x1, y0, y1, z0, z1 = min(xa, xb), max(xa, xb), min(ya, yb), max(ya, yb), min(za, zb), max(za, zb)
    return obox(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (1, 0, 0), (0, 1, 0), x1 - x0, y1 - y0, z1 - z0)


def obox(centre, xdir, ydir, lx, ly, lz):
    """A box with its length along xdir, width along ydir and height along xdir x ydir."""
    return tmp.createBox(adsk.core.OrientedBoundingBox3D.create(P(*centre), V(*xdir), V(*ydir), lx * MM, ly * MM, lz * MM))


def cyl(p1, p2, r1, r2=None):
    return tmp.createCylinderOrCone(P(*p1), r1 * MM, P(*p2), (r1 if r2 is None else r2) * MM)


def cyl_y(x, z, ya, yb, r):
    return cyl((x, min(ya, yb), z), (x, max(ya, yb), z), r)


def cyl_x(y, z, xa, xb, r):
    return cyl((min(xa, xb), y, z), (max(xa, xb), y, z), r)


def cyl_z(x, y, za, zb, r, r2=None):
    return cyl((x, y, za), (x, y, zb), r, r2)


def sphere(c, r):
    return tmp.createSphere(P(*c), r * MM)


def torus(c, axis, major, minor):
    return tmp.createTorus(P(*c), V(*axis), major * MM, minor * MM)


def cut(a, b):
    tmp.booleanOperation(a, b, adsk.fusion.BooleanTypes.DifferenceBooleanType)
    return a


def join(a, b):
    tmp.booleanOperation(a, b, adsk.fusion.BooleanTypes.UnionBooleanType)
    return a


def common(a, b):
    tmp.booleanOperation(a, b, adsk.fusion.BooleanTypes.IntersectionBooleanType)
    return a


def copy(b):
    return tmp.copy(b)


def moved(b, matrix):
    tmp.transform(b, matrix)
    return b


def frame(origin, x, y, z):
    m = adsk.core.Matrix3D.create()
    m.setWithCoordinateSystem(P(*origin), V(*x), V(*y), V(*z))
    return m


def shifted(b, dx, dy, dz):
    m = adsk.core.Matrix3D.create()
    m.translation = V(dx * MM, dy * MM, dz * MM)
    return moved(b, m)


def _fillet(body, edges, r):
    fillets = root.features.filletFeatures
    chosen = adsk.core.ObjectCollection.create()
    for e in edges:
        chosen.add(e)
    spec = fillets.createInput()
    spec.isRollingBallCorner = True
    spec.edgeSetInputs.addConstantRadiusEdgeSet(chosen, adsk.core.ValueInput.createByReal(r * MM), True)
    return fillets.add(spec).bodies.item(0)


def _flat_face_edges(body, z):
    for f in body.faces:
        g = f.geometry
        if g.surfaceType == adsk.core.SurfaceTypes.PlaneSurfaceType and abs(abs(g.normal.z) - 1) < 1e-6 \
                and abs(f.pointOnFace.z - z * MM) < 1e-6:
            return list(f.edges)
    raise RuntimeError(f"no flat face at z={z}")


def rbox(xa, xb, ya, yb, za, zb, rv=0.0, rt=0.0, rb=0.0):
    """A box with rounded upright edges (rv) and rounded top (rt) and bottom (rb) rims."""
    body = root.bRepBodies.add(box(xa, xb, ya, yb, za, zb))
    if rv:
        upright = [e for e in body.edges if abs(e.startVertex.geometry.x - e.endVertex.geometry.x) < 1e-6
                   and abs(e.startVertex.geometry.y - e.endVertex.geometry.y) < 1e-6]
        body = _fillet(body, upright, rv)
    if rt:
        body = _fillet(body, _flat_face_edges(body, max(za, zb)), rt)
    if rb:
        body = _fillet(body, _flat_face_edges(body, min(za, zb)), rb)
    out = tmp.copy(body)
    body.deleteMe()
    return out


def rplate(width, height, thickness, r, origin, x, y, z):
    """A rounded plate, width along x and height along y, growing by its thickness along z from the origin."""
    return moved(rbox(-width / 2, width / 2, -height / 2, height / 2, 0, thickness, rv=r), frame(origin, x, y, z))


# ------------------------------------------------------------------ document helpers
def look(name, base, rgb=None):
    if name not in _looks:
        made = design.appearances.itemByName(name)
        if not made:
            made = design.appearances.addByCopy(lib.appearances.itemByName(base), name)
        if rgb:
            colours = [adsk.core.ColorProperty.cast(p) for p in made.appearanceProperties]
            colours = [c for c in colours if c]
            for c in [c for c in colours if c.name == "Color"] or colours[:1]:
                c.value = adsk.core.Color.create(rgb[0], rgb[1], rgb[2], 255)
        _looks[name] = made
    return _looks[name]


def part(name, parent=None):
    occ = (parent.component if parent else root).occurrences.addNewComponent(adsk.core.Matrix3D.create())
    occ.component.name = name
    return occ


def put(occ, body, name, appearance):
    made = occ.component.bRepBodies.add(body)
    made.name = name
    made.appearance = appearance
    return made


# ------------------------------------------------------------------ the robot
def build():
    global app, design, root, tmp, lib
    app = adsk.core.Application.get()
    tmp = adsk.fusion.TemporaryBRepManager.get()
    lib = app.materialLibraries.itemByName("Fusion Appearance Library")
    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design or not design.attributes.itemByName("sprocket", "generated"):
        app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
        design = adsk.fusion.Design.cast(app.activeProduct)
        design.designType = adsk.fusion.DesignTypes.DirectDesignType
        design.attributes.add("sprocket", "generated", "1")
    root = design.rootComponent
    for occ in list(root.occurrences):
        occ.deleteMe()
    for b in list(root.bRepBodies):
        b.deleteMe()
    _looks.clear()

    white = look("Sprocket Shell White", "Plastic - Matte (White)", (236, 234, 228))
    graphite = look("Sprocket Graphite", "Plastic - Matte (Black)", (96, 101, 110))
    rubber = look("Sprocket Rubber", "Plastic - Matte (Black)", (60, 60, 64))
    dark_metal = look("Sprocket Dark Anodized", "Plastic - Matte (Black)", (84, 88, 96))
    metal = look("Sprocket Aluminium", "Aluminum - Satin")
    orange = look("Sprocket Orange", "Plastic - Glossy (Yellow)", (255, 118, 20))
    visor_black = look("Sprocket Visor", "Plastic - Glossy (Black)", (10, 11, 14))
    lens = look("Sprocket Lens", "Plastic - Glossy (Black)", (18, 30, 54))
    glow = look("Sprocket Eye Glow", "LED (Blue)", (80, 225, 255))
    headlight = look("Sprocket Headlight", "LED - SMD 5050 - 20lm (White)")
    tail = look("Sprocket Tail Light", "LED (Red)")
    brass = look("Sprocket Brass", "Brass - Polished")
    board_teal = look("Sprocket Board Teal", "Plastic - Matte (Green)", (0, 124, 130))
    board_green = look("Sprocket Board Green", "Plastic - Matte (Green)", (24, 112, 64))
    cell_blue = look("Sprocket Battery Blue", "Plastic - Matte (Blue)", (38, 92, 205))
    module = look("Sprocket Module Grey", "Plastic - Matte (Gray)", (128, 132, 138))
    stop_red = look("Sprocket Stop Red", "Plastic - Glossy (Red)", (214, 26, 26))
    stop_yellow = look("Sprocket Stop Yellow", "Plastic - Glossy (Yellow)", (250, 205, 20))

    corners = {"FL": (1, 1), "FR": (1, -1), "RL": (-1, 1), "RR": (-1, -1)}

    def shock_ends(s, t):
        return (s * SHOCK_LO[0], t * SHOCK_Y, SHOCK_LO[1]), (s * SHOCK_HI[0], t * SHOCK_Y, SHOCK_HI[1])

    # ---------------------------------------------------------------- chassis
    chassis = part("Chassis")
    h_out = TUB_CHAMFER / math.sqrt(2)
    h_in = h_out + TUB_WALL
    tub = rbox(-TUB_X, TUB_X, -TUB_Y, TUB_Y, TUB_Z0, TUB_Z1, rv=8)
    hollow = rbox(-TUB_X + TUB_WALL, TUB_X - TUB_WALL, -TUB_Y + TUB_WALL, TUB_Y - TUB_WALL, TUB_Z0 + TUB_WALL,
                  TUB_Z1 + 1, rv=5)
    for s in (1, -1):  # sloped nose and tail for the approach and departure angles
        along = unit((s, 0, 1))
        cut(tub, obox((s * TUB_X, 0, TUB_Z0), along, (0, 1, 0), 6 * h_out, 400, 2 * h_out))
        cut(hollow, obox((s * TUB_X, 0, TUB_Z0), along, (0, 1, 0), 8 * h_in, 400, 2 * h_in))
    cut(tub, hollow)
    put(chassis, tub, "Tub (3 mm aluminium)", dark_metal)

    deck = rbox(-DECK_X, DECK_X, -DECK_Y, DECK_Y, DECK_Z0, DECK_Z1, rv=DECK_R)
    for s, t in corners.values():
        lo, hi = shock_ends(s, t)
        u = unit(add3(hi, lo, -1))
        cut(deck, cyl(add3(lo, u, 30), add3(lo, u, 62), 13))
    cut(deck, box(62, 98, -12, 12, DECK_Z0 - 1, DECK_Z1 + 1))  # power and motor-bus cables up from the tub
    put(chassis, deck, "Deck plate (4 mm aluminium)", metal)

    bumper = rbox(TUB_X, TUB_X + 14, -80, 80, 62, 92, rv=5, rt=4, rb=4)
    put(chassis, copy(bumper), "Front bumper", rubber)
    put(chassis, shifted(bumper, -2 * TUB_X - 14, 0, 0), "Rear bumper", rubber)
    for t in (1, -1):
        put(chassis, cyl_x(t * 50, 77, TUB_X + 14, TUB_X + 14.8, 5), "Ground ToF window", visor_black)
        put(chassis, box(-TUB_X - 15, -TUB_X - 14, t * 25 - 6, t * 25 + 6, 67, 87), "Dock contact", brass)

    for name, (s, t) in corners.items():
        lo, hi = shock_ends(s, t)
        tower = box(hi[0] - 12, hi[0] + 12, t * 86, t * 90, DECK_Z1, 170)
        join(tower, box(hi[0] - 12, hi[0] + 12, t * 66, t * 88, DECK_Z1, DECK_Z1 + 3))
        cut(tower, cyl_y(hi[0], 142, t * 85, t * 91, 6))
        join(tower, cyl_y(hi[0], hi[2], t * 86, t * 108, 2))  # the stud the shock's upper eye hangs on
        put(chassis, tower, f"Shock tower {name}", dark_metal)

    # ---------------------------------------------------------------- wheels and suspension
    tyre = cyl((0, -TYRE_W / 2, 0), (0, TYRE_W / 2, 0), WHEEL_R)
    common(tyre, sphere((0, 0, 0), WHEEL_R + 2.5))  # crowned tread
    cut(tyre, cyl((0, -TYRE_W / 2 - 1, 0), (0, TYRE_W / 2 + 1, 0), HUB_R))
    for i in range(24):  # chevron tread
        spin = adsk.core.Matrix3D.create()
        spin.setToRotation(2 * math.pi * i / 24, V(0, 1, 0), P(0, 0, 0))
        for side in (1, -1):
            d = unit((-8.0, side * 21.0, 0.0))
            groove = obox((d[0] * 11.5, d[1] * 11.5, WHEEL_R - 0.5), d, (-d[1], d[0], 0), 25, 4, 8)
            cut(tyre, moved(groove, spin))
    hub = cyl((0, -TYRE_W / 2 + 1, 0), (0, TYRE_W / 2 - 1, 0), HUB_R)
    for i in range(5):
        a = 2 * math.pi * i / 5
        for side in (1, -1):
            y = side * (TYRE_W / 2 - 1)
            cut(hub, cyl((24 * math.cos(a), y - side * 3, 24 * math.sin(a)), (24 * math.cos(a), y, 24 * math.sin(a)), 7))

    for name, (s, t) in corners.items():
        corner = part(f"Suspension {name}")
        axle, pivot = (s * AX, t * 99, AZ), (s * PIVOT_X, t * 99, PIVOT_Z)
        lo, hi = shock_ends(s, t)
        u = unit(add3(hi, lo, -1))
        reach = math.dist(axle, pivot)
        mid = add3(axle, add3(pivot, axle, -1), 0.5)
        link_dir = unit(add3(axle, pivot, -1))

        arm = cyl_y(axle[0], axle[2], t * ARM_Y0, t * ARM_Y1, 19)
        join(arm, cyl_y(pivot[0], pivot[2], t * ARM_Y0, t * ARM_Y1, 15))
        join(arm, obox(mid, link_dir, (0, 1, 0), reach, ARM_Y1 - ARM_Y0, 26))
        cut(arm, obox((mid[0], t * (ARM_Y1 - 1.4), mid[2]), link_dir, (0, 1, 0), 40, 3.2, 12))  # machined pocket
        join(arm, cyl_y(pivot[0], pivot[2], t * TUB_Y, t * ARM_Y0, 11))  # pivot bearing housing
        join(arm, box(lo[0] - 6, lo[0] + 6, t * ARM_Y0, t * (ARM_Y0 + 4), 60, 83))  # shock lug
        join(arm, cyl_y(lo[0], lo[2], t * ARM_Y0, t * 107, 2))
        put(corner, arm, "Trailing arm (7075)", dark_metal)
        put(corner, cyl_y(pivot[0], pivot[2], t * ARM_Y1, t * (ARM_Y1 + 3), 6), "Pivot bolt", metal)
        put(corner, cyl_y(axle[0], axle[2], t * ARM_Y1, t * (AY - TYRE_W / 2 + 1), 7), "Motor stub axle", metal)

        put(corner, shifted(copy(tyre), s * AX, t * AY, AZ), "Tyre 100 mm", rubber)
        put(corner, shifted(copy(hub), s * AX, t * AY, AZ), "Hub motor DDSM115", dark_metal)
        y_cap = t * (AY + TYRE_W / 2 - 1)
        put(corner, cyl_y(s * AX, AZ, y_cap, y_cap + t * 2.5, 15), "Hub cap", orange)

        shock = cyl(add3(lo, u, 3), add3(lo, u, 50), 2)  # rod
        join(shock, cyl(add3(hi, u, -50), add3(hi, u, -8), 7))  # damper body
        join(shock, cyl(add3(hi, u, -8), add3(hi, u, -3), 8))
        join(shock, cyl(add3(hi, u, -16), add3(hi, u, -13), 11))  # preload collar
        join(shock, cyl(add3(lo, u, 12), add3(lo, u, 14.5), 11))  # lower spring seat
        for eye in (lo, hi):
            join(shock, cyl_y(eye[0], eye[2], t * 98, t * 106, 5))
            cut(shock, cyl_y(eye[0], eye[2], t * 90, t * 110, 2))
        put(corner, shock, "Coil-over damper 90 mm", metal)

        start, end, coils, radius = 17.5, 70.5, 7, 9.0
        lean = math.atan((end - start) / coils / (2 * math.pi * radius))
        axis = unit(add3((u[0] * math.cos(lean), 0, u[2] * math.cos(lean)), (0, 1, 0), math.sin(lean)))
        spring = None
        for i in range(coils + 1):
            ring = torus(add3(lo, u, start + (end - start) * i / coils), axis, radius, 1.5)
            spring = ring if spring is None else join(spring, ring)
        put(corner, spring, "Spring", orange)

    # ---------------------------------------------------------------- body shell
    body = part("Body shell")
    shell = rbox(-SHELL_X, SHELL_X, -SHELL_Y, SHELL_Y, SHELL_Z0, SHELL_Z1, rv=SHELL_RV, rt=SHELL_RT)
    w = SHELL_WALL
    cut(shell, rbox(-SHELL_X + w, SHELL_X - w, -SHELL_Y + w, SHELL_Y - w, SHELL_Z0 - 3, SHELL_Z1 - w,
                    rv=SHELL_RV - w, rt=SHELL_RT - w))
    cut(shell, cyl_z(NECK_X, 0, SHELL_Z1 - 8, SHELL_Z1 + 2, 16))  # cables up the neck
    hatch = rbox(-106, 36, -88, 88, SHELL_Z1 - 1.2, SHELL_Z1, rv=18)  # lift-off cover over the computer
    cut(shell, box(-102, 32, -84, 84, SHELL_Z1 - 8, SHELL_Z1 + 2))
    cut(shell, copy(hatch))
    cut(shell, cyl_z(-126, 0, SHELL_Z1 - 8, SHELL_Z1 + 2, 11.5))  # emergency stop
    for dx in (-30, 30):
        for dy in (-30, 30):
            cut(hatch, cyl_z(-55 + dx, dy, SHELL_Z1 - 8, SHELL_Z1 + 2, 1.5))  # microphone ports
    put(body, hatch, "Service hatch", graphite)
    for z in (137, 143, 149):
        for k in range(6):
            cut(shell, cyl_x(-15 + 6 * k, z, SHELL_X - 10, SHELL_X + 5, 1.6))  # speaker grille
    for t in (1, -1):
        for k in range(6):
            x = -100 + 12 * k
            cut(shell, box(x - 2, x + 2, t * (SHELL_Y - 6), t * (SHELL_Y + 2), 140, 160))  # cooling vents
    cut(shell, box(SHELL_X - 1.2, SHELL_X + 1, -55, 55, 162, 170))  # headlight bar
    cut(shell, box(-SHELL_X + 1.2, -SHELL_X - 1, -40, 40, 162, 168))  # tail light
    cut(shell, cyl_x(-35, 143, -SHELL_X - 2, -SHELL_X + 6, 6.5))  # charge socket
    cut(shell, cyl_x(35, 143, -SHELL_X - 2, -SHELL_X + 6, 8.5))  # power button
    stripe = common(copy(shell), box(-200, 200, -200, 200, 124, 127))
    cut(shell, box(-200, 200, -200, 200, 124, 127))
    put(body, shell, "Shell (3 mm ASA)", white)
    put(body, stripe, "Belt line", orange)
    put(body, box(SHELL_X - 1.2, SHELL_X + 0.3, -55, 55, 162, 170), "Headlight bar", headlight)
    put(body, box(-SHELL_X + 1.2, -SHELL_X - 0.3, -40, 40, 162, 168), "Tail light", tail)
    put(body, cyl_x(-35, 143, -SHELL_X - 0.5, -SHELL_X + 10, 6.5), "Charge socket", graphite)
    put(body, cyl_x(35, 143, -SHELL_X - 2, -SHELL_X + 10, 7), "Power button", metal)
    ring = cut(cyl_x(35, 143, -SHELL_X - 0.6, -SHELL_X + 1, 8.5), cyl_x(35, 143, -SHELL_X - 3, -SHELL_X + 3, 7))
    put(body, ring, "Power button ring", glow)
    put(body, cyl_z(-126, 0, SHELL_Z1, SHELL_Z1 + 4, 17), "E-stop collar", stop_yellow)
    stop = join(cyl_z(-126, 0, SHELL_Z1 + 4, SHELL_Z1 + 14, 15), cyl_z(-126, 0, SHELL_Z1 + 14, SHELL_Z1 + 18, 15, 11))
    put(body, stop, "E-stop button", stop_red)

    # ---------------------------------------------------------------- electronics
    parts = part("Electronics")
    put(parts, box(-112, 48, -50, 50, 130, 156), "Arduino VENTUNO Q (160 x 100 x 25.8)", board_teal)
    put(parts, box(-60, -15, -22.5, 22.5, 156, 170), "Heatsink and fan", module)
    for x in (-108, 44):
        for y in (-46, 46):
            put(parts, cyl_z(x, y, DECK_Z1, 130, 3), "Standoff", metal)
    put(parts, box(-12, 12, -10, 10, DECK_Z1, DECK_Z1 + 5), "IMU (BNO085)", board_green)
    put(parts, box(-60, 0, 60, 96, DECK_Z1 + 2, DECK_Z1 + 20), "DC-DC 5 V 8 A", module)
    put(parts, box(-50, -5, -92, -62, DECK_Z1 + 2, DECK_Z1 + 14), "RS-485 motor bus adapter", board_green)
    put(parts, cyl_z(-55, 0, 180, 188, 50), "Microphone array (XVF3800, 4 mics)", board_green)
    put(parts, cyl_z(-126, 0, 152, SHELL_Z1, 11), "E-stop switch body", graphite)
    speaker = join(cyl_x(0, 143, 136, 154, 18), cyl_x(0, 143, 126, 136, 10))
    put(parts, speaker, "Speaker 36 mm 5 W", graphite)
    put(parts, box(-70, 50, -39, 39, 49, 99), "Battery 5S2P 21700, 18 V 10 Ah", cell_blue)
    put(parts, box(55, 95, -45, 45, 49, 79), "Power board (BMS, fuses, e-stop contactor)", module)
    put(parts, box(-105, -75, -30, 30, 60, 75), "Motor bus and power hub", board_green)

    # ---------------------------------------------------------------- head (built in its own frame, then tilted)
    tilt = adsk.core.Matrix3D.create()
    tilt.setToRotation(math.radians(-HEAD_TILT), V(0, 1, 0), P(0, 0, 0))
    tilt.translation = V(NECK_X * MM, 0, HEAD_PIVOT_Z * MM)
    head = root.occurrences.addNewComponent(tilt)
    head.component.name = "Head"

    solid = rbox(HEAD_X0, HEAD_X1, -HEAD_Y, HEAD_Y, 0, HEAD_H, rv=HEAD_RV, rt=HEAD_R, rb=HEAD_R)
    head_in_world = moved(copy(solid), tilt)
    w = HEAD_WALL
    skull = cut(solid, rbox(HEAD_X0 + w, HEAD_X1 - w, -HEAD_Y + w, HEAD_Y - w, w, HEAD_H - w,
                            rv=HEAD_RV - w, rt=HEAD_R - w, rb=HEAD_R - w))
    face = dict(x=(0, 1, 0), y=(0, 0, 1), z=(1, 0, 0))
    cut(skull, rplate(114, 70, 5, 12, (HEAD_X1 - 1.5, 0, 52), **face))  # visor seat
    cut(skull, box(HEAD_X1 - 4, HEAD_X1, -37, 37, 19.5, 68.5))  # display window
    cut(skull, cyl_x(0, 80, HEAD_X1 - 8, HEAD_X1, 6.5))  # front camera
    cam_yaw = math.radians(REAR_CAMERA_YAW)
    rear = []
    for t in (1, -1):
        d = (math.cos(cam_yaw), t * math.sin(cam_yaw), 0)
        centre = (HEAD_X0 + HEAD_RV, t * (HEAD_Y - HEAD_RV), 62)
        cut(skull, cyl(add3(centre, d, 15), add3(centre, d, 30), 9))
        rear.append((t, d, centre))
    put(head, skull, "Head shell (2.5 mm ASA)", white)
    put(head, rplate(114, 70, 2.1, 12, (HEAD_X1 - 1.5, 0, 52), **face), "Visor (tinted PMMA)", visor_black)
    for t in (1, -1):
        put(head, rplate(16, 26, 0.4, 7.9, (HEAD_X1 + 0.6, t * 24, 44), **face), "Eye", glow)
    put(head, cyl_x(0, 80, HEAD_X1 + 0.6, HEAD_X1 + 0.9, 5.5), "Front camera window", lens)
    put(head, box(HEAD_X1 - 9, HEAD_X1 - 3, -46, 46, 16, 72), "Display 3.5 in IPS 480 x 320", graphite)
    put(head, box(HEAD_X1 - 17, HEAD_X1 - 14, -12.5, 12.5, 66, 90), "Front camera board (CSI)", board_green)
    put(head, cyl_x(0, 80, HEAD_X1 - 14, HEAD_X1 - 1.5, 6), "Front camera lens", graphite)
    for t, d, centre in rear:
        side = "left" if t > 0 else "right"
        across = (-d[1], d[0], 0)
        put(head, obox(add3(centre, d, 12.75), across, (0, 0, 1), 25, 24, 1.5), f"Rear {side} camera board (CSI)",
            board_green)
        put(head, cyl(add3(centre, d, 13.5), add3(centre, d, 19.5), 6), f"Rear {side} camera lens", graphite)
        put(head, cyl(add3(centre, d, 19.5), add3(centre, d, 23), 9), f"Rear {side} camera bezel", visor_black)
        put(head, cyl(add3(centre, d, 23), add3(centre, d, 23.6), 5), f"Rear {side} camera window", lens)

    # ---------------------------------------------------------------- neck
    neck = part("Neck")
    column = join(cyl_z(NECK_X, 0, SHELL_Z1 + 3, HEAD_PIVOT_Z - 4, 27, 22), sphere((NECK_X, 0, HEAD_PIVOT_Z - 1), 25))
    join(column, cyl_z(NECK_X, 0, SHELL_Z1, SHELL_Z1 + 3, 32))
    cut(column, cyl_z(NECK_X, 0, SHELL_Z1 - 1, HEAD_PIVOT_Z + 30, 16))
    cut(column, copy(head_in_world))
    put(neck, column, "Neck", graphite)
    halo = cut(cyl_z(NECK_X, 0, SHELL_Z1 + 3, SHELL_Z1 + 7, 31), cyl_z(NECK_X, 0, SHELL_Z1 + 2, SHELL_Z1 + 8, 28))
    put(neck, halo, "Status light ring", glow)

    # ---------------------------------------------------------------- lidar (level, on top of the tilted head)
    top = P(9, 0, HEAD_H)
    top.transformBy(tilt)
    lx = round(top.x / MM)
    lidar = part("Lidar")
    pedestal = cut(cyl_z(lx, 0, LIDAR_TOP_OF_PEDESTAL - 29, LIDAR_TOP_OF_PEDESTAL, 32), copy(head_in_world))
    put(lidar, pedestal, "Lidar pedestal (level)", white)
    z = LIDAR_TOP_OF_PEDESTAL
    put(lidar, cyl_z(lx, 0, z, z + 21, 28), "Lidar base (RPLIDAR C1)", graphite)
    put(lidar, cyl_z(lx, 0, z + 21, z + 41, 25.5), "Lidar scanner window", visor_black)
    put(lidar, cyl_z(lx, 0, z + 41, z + 42.5, 20), "Lidar cap", graphite)

    box3 = root.boundingBox
    size = [round((box3.maxPoint.asArray()[i] - box3.minPoint.asArray()[i]) / MM) for i in range(3)]
    bodies = sum(o.bRepBodies.count for o in root.allOccurrences)
    print(f"Sprocket built: {size[0]} x {size[1]} x {size[2]} mm (L x W x H), {bodies} bodies, lidar at x={lx}")
