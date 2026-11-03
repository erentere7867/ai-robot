// Real-time base controller for the Arduino VENTUNO Q's STM32H5 (Cortex-M33, Arduino core on Zephyr).
//
// The Qualcomm side (Linux) thinks and plans; this MCU owns everything that must happen on time, whatever
// Linux is doing:
//   * 200 Hz wheel velocity loops: feed-forward + PI with anti-windup, filtered encoder speed, accel limits
//   * odometry at 200 Hz: distance from the encoders, heading from the gyro (skid steer slips in turns);
//     the gyro bias is re-learned whenever the robot stands still
//   * safety envelope: forward speed is capped to what can still stop before the range-sensor reading
//     (v <= sqrt(2 a d)), so the robot slows down smoothly instead of slamming to a halt; bumper, cliff,
//     lifted, stall and emergency stop latch the motors off until the host commands a way out
//   * command watchdog: no velocity command for 300 ms -> stop
//   * 50 Hz telemetry to Linux: pose, velocities, raw ticks, range, fault flags, battery
//
// Link: Arduino Router Bridge (msgpack-RPC; the Linux side is robot/body/mcu_body.py over
// /var/run/arduino-router.sock). Methods this sketch provides, all as notifications (no reply):
//   cmd_vel(v_mm_s, w_mdeg_s)   head(tilt_ddeg)   stop()   estop(on)   reset_odom()
// It notifies `telemetry` with:
//   [ms, x_mm, y_mm, heading_mdeg, v_mm_s, w_mdeg_s, ticks_l, ticks_r, range_mm, flags, battery_mv, head_ddeg]
// flags: 1 bumper, 2 cliff, 4 estop, 8 picked up, 16 charging, 32 forward blocked, 64 wheel stall,
//        128 IMU fault, 256 range sensor fault, 512 battery low
//
// Hardware as config/body.yaml: TB6612FNG dual driver (left motors on A, right on B), one quadrature
// encoder per side, VL53L1X range sensor, an IMU with a Z gyro (MPU6050 here), bumpers, four IR cliff
// sensors, head servo, latching e-stop, battery divider. The PIN_* numbers are placeholders: map them to
// the VENTUNO Q MCU header you wire to. Libraries: Arduino_RouterBridge, Pololu VL53L1X, Adafruit MPU6050.

#include <Arduino_RouterBridge.h>
#include <Wire.h>
#include <Servo.h>
#include <VL53L1X.h>
#include <Adafruit_MPU6050.h>

// ---- wiring (adapt) ------------------------------------------------------------------------------
const int PIN_L_PWM = 3, PIN_L_IN1 = 4, PIN_L_IN2 = 5;
const int PIN_R_PWM = 6, PIN_R_IN1 = 7, PIN_R_IN2 = 8;
const int PIN_STBY = 9;
const int PIN_ENC_LA = 2, PIN_ENC_LB = 10, PIN_ENC_RA = 11, PIN_ENC_RB = 12;
const int PIN_BUMP_L = A0, PIN_BUMP_R = A1;            // microswitches to GND (active low)
const int PIN_CLIFF[4] = {A2, A3, 13, 14};               // IR floor sensors, LOW = floor seen
const int PIN_SERVO = 15;
const int PIN_ESTOP = 16;                                // latching button to GND
const int PIN_BATT = A4;                                 // through a divider
const int PIN_CHARGE = 17;                               // dock contacts sense, HIGH = charging
const float BATT_DIVIDER = 3.0f, ADC_VREF = 3.3f, ADC_MAX = 4095.0f;

// ---- robot (as config/body.yaml) -------------------------------------------------------------------
const float WHEEL_DIAMETER_MM = 120.0f, TRACK_MM = 208.0f, TURN_SLIP = 2.4f;
const float TICKS_PER_REV = 720.0f;
const float MM_PER_TICK = 3.14159265f * WHEEL_DIAMETER_MM / TICKS_PER_REV;
const float MAX_V = 400.0f, MAX_W = 120.0f;              // mm/s, deg/s
const float ACCEL = 900.0f, DECEL = 1200.0f, ANG_ACCEL = 600.0f;
const float STOP_MM = 110.0f;                            // range reading at which forward motion must be 0
const float BRAKE_DECEL = 700.0f;                        // mm/s^2 used for the safety envelope (conservative)
const float BATT_LOW_MV = 6600.0f;

// ---- timing --------------------------------------------------------------------------------------
const uint32_t CONTROL_US = 5000;                        // 200 Hz
const uint32_t TELEMETRY_MS = 20;                        // 50 Hz
const uint32_t WATCHDOG_MS = 300;
const uint32_t RANGE_TIMEOUT_MS = 200;

// ---- loop gains (per side, mm/s error -> PWM fraction) --------------------------------------------
const float KFF = 1.0f / (MAX_V * 1.2f), KP = 0.0025f, KI = 0.02f, KAW = 2.0f;
const float SPEED_LPF = 0.35f;                           // encoder speed low-pass (per control tick)

volatile int32_t ticksL = 0, ticksR = 0;
VL53L1X tof;
Adafruit_MPU6050 imu;
Servo head;
bool imuOk = false, tofOk = false;

// commands (written by Bridge callbacks, read by the control loop; provide_safe runs them in loop context)
float targetV = 0, targetW = 0;
uint32_t lastCmdMs = 0;
bool estopCmd = false, latched = false;
int headDdeg = 0;

// state
float cmdV = 0, cmdW = 0;                                // ramped set point
float speedL = 0, speedR = 0, integL = 0, integR = 0;    // measured wheel speeds, integrators
float dutyL = 0, dutyR = 0;
int32_t lastTicksL = 0, lastTicksR = 0, odomTicksL = 0, odomTicksR = 0;
double odomX = 0, odomY = 0, odomTh = 0;                 // mm, mm, rad
float gyroZ = 0, gyroBias = 0;                           // deg/s
float measV = 0, measW = 0;
uint32_t stillSinceMs = 0, stallSinceMs = 0, lastRangeMs = 0;
float biasAcc = 0;
int biasN = 0;
int rangeMm = 2000;
uint32_t lastControlUs = 0, lastTelemetryMs = 0;
int faults = 0;

void encL() { ticksL += (digitalRead(PIN_ENC_LA) == digitalRead(PIN_ENC_LB)) ? 1 : -1; }
void encR() { ticksR += (digitalRead(PIN_ENC_RA) == digitalRead(PIN_ENC_RB)) ? -1 : 1; }

void setMotor(int pwmPin, int in1, int in2, float duty) {
  duty = constrain(duty, -1.0f, 1.0f);
  digitalWrite(in1, duty > 0);
  digitalWrite(in2, duty < 0);
  analogWrite(pwmPin, (int)(fabsf(duty) * 255));
}

void motorsOff() {
  setMotor(PIN_L_PWM, PIN_L_IN1, PIN_L_IN2, 0);
  setMotor(PIN_R_PWM, PIN_R_IN1, PIN_R_IN2, 0);
  integL = integR = dutyL = dutyR = 0;
}

void haltAll() {
  targetV = targetW = cmdV = cmdW = 0;
  motorsOff();
}

// ---- Bridge methods (Linux -> MCU) ------------------------------------------------------------------
void onCmdVel(int v, int wMdeg) {
  targetV = constrain((float)v, -MAX_V, MAX_V);
  targetW = constrain(wMdeg / 1000.0f, -MAX_W, MAX_W);
  lastCmdMs = millis();
  // Backing away or turning in place is the way out of a contact: it clears the latch.
  if (targetV <= 0) latched = false;
}
void onHead(int ddeg) {
  headDdeg = constrain(ddeg, -250, 350);
  head.write(constrain(90 + headDdeg / 10, 60, 130));
}
void onStop() { haltAll(); latched = false; }
void onEstop(bool on) { estopCmd = on; if (on) haltAll(); }
void onResetOdom() { odomX = odomY = odomTh = 0; }

// ---- sensing ----------------------------------------------------------------------------------------
int readFlags() {
  int f = 0;
  if (!digitalRead(PIN_BUMP_L) || !digitalRead(PIN_BUMP_R)) f |= 1;
  int floorless = 0;
  for (int i = 0; i < 4; i++) floorless += digitalRead(PIN_CLIFF[i]) ? 1 : 0;
  if (floorless == 1 || floorless == 2) f |= 2;         // one or two edges: a drop ahead or behind
  if (floorless >= 3) f |= 8;                           // nearly all see nothing: picked up
  if (!digitalRead(PIN_ESTOP) || estopCmd) f |= 4;
  if (digitalRead(PIN_CHARGE)) f |= 16;
  if (targetV > 0 && rangeMm <= STOP_MM) f |= 32;
  return f | faults;
}

float batteryMv() { return analogRead(PIN_BATT) * ADC_VREF / ADC_MAX * BATT_DIVIDER * 1000.0f; }

// ---- control ----------------------------------------------------------------------------------------
void control(float dt) {
  uint32_t nowMs = millis();
  if (imuOk) {
    sensors_event_t a, g, t;
    imu.getEvent(&a, &g, &t);
    gyroZ = g.gyro.z * 57.29578f - gyroBias;
  }
  // Wheel speeds from the encoders, low-passed (a 720 tick/rev encoder at 200 Hz is coarse).
  int32_t tl = ticksL, tr = ticksR;
  float rawL = (tl - lastTicksL) * MM_PER_TICK / dt, rawR = (tr - lastTicksR) * MM_PER_TICK / dt;
  lastTicksL = tl;
  lastTicksR = tr;
  speedL += SPEED_LPF * (rawL - speedL);
  speedR += SPEED_LPF * (rawR - speedR);

  // Odometry: distance from the wheels, heading from the gyro (falls back to the wheels without an IMU).
  float dl = (tl - odomTicksL) * MM_PER_TICK, dr = (tr - odomTicksR) * MM_PER_TICK;
  odomTicksL = tl;
  odomTicksR = tr;
  float ds = 0.5f * (dl + dr);
  float dth = imuOk ? gyroZ * 0.0174533f * dt : (dr - dl) / (TRACK_MM * TURN_SLIP);
  double mid = odomTh + 0.5 * dth;
  odomX += ds * cos(mid);
  odomY += ds * sin(mid);
  odomTh = atan2(sin(odomTh + dth), cos(odomTh + dth));
  measV = ds / dt;
  measW = dth / dt * 57.29578f;

  // Re-learn the gyro bias while standing still (wheels idle and not commanded for a second).
  bool still = fabsf(speedL) < 2 && fabsf(speedR) < 2 && targetV == 0 && targetW == 0;
  if (!still) { stillSinceMs = nowMs; biasAcc = 0; biasN = 0; }
  else if (imuOk && nowMs - stillSinceMs > 1000) {
    biasAcc += gyroZ;
    if (++biasN >= 200) { gyroBias += biasAcc / biasN; biasAcc = 0; biasN = 0; }
  }

  // Faults and reflexes.
  faults = 0;
  if (!imuOk) faults |= 128;
  if (!tofOk || nowMs - lastRangeMs > RANGE_TIMEOUT_MS) faults |= 256;
  if (batteryMv() < BATT_LOW_MV) faults |= 512;
  bool pushing = fabsf(dutyL) > 0.6f || fabsf(dutyR) > 0.6f;
  bool notTurning = fabsf(speedL) < 15 && fabsf(speedR) < 15;
  if (!(pushing && notTurning)) stallSinceMs = nowMs;
  else if (nowMs - stallSinceMs > 600) faults |= 64;     // driving hard but the wheels do not move
  int f = readFlags();
  if (f & (1 | 2 | 4 | 8 | 64)) latched = true;

  float v = targetV, w = targetW;
  if (nowMs - lastCmdMs > WATCHDOG_MS || latched || (f & 4)) v = w = 0;
  // Safety envelope: never faster forward than can still stop before the obstacle the range sensor sees.
  // A missing range reading counts as an obstacle at the stop distance (creep only).
  float clear = (faults & 256) ? 0.0f : max(0.0f, (float)rangeMm - STOP_MM);
  if (v > 0) v = min(v, sqrtf(2.0f * BRAKE_DECEL * clear));
  float accV = (fabsf(v) < fabsf(cmdV) || v * cmdV < 0) ? DECEL : ACCEL;
  cmdV += constrain(v - cmdV, -accV * dt, accV * dt);
  cmdW += constrain(w - cmdW, -ANG_ACCEL * dt, ANG_ACCEL * dt);
  if (v == 0 && w == 0 && fabsf(cmdV) < 1 && fabsf(cmdW) < 1) {
    cmdV = cmdW = 0;
    motorsOff();
    return;
  }
  float turn = cmdW * 0.0174533f * TRACK_MM * 0.5f * TURN_SLIP;
  float wantL = cmdV - turn, wantR = cmdV + turn;
  float uL = wantL * KFF + KP * (wantL - speedL) + KI * integL;
  float uR = wantR * KFF + KP * (wantR - speedR) + KI * integR;
  float sL = constrain(uL, -1.0f, 1.0f), sR = constrain(uR, -1.0f, 1.0f);
  // Back-calculation anti-windup: a saturated output bleeds its integrator.
  integL += ((wantL - speedL) + KAW * (sL - uL) / KI) * dt;
  integR += ((wantR - speedR) + KAW * (sR - uR) / KI) * dt;
  dutyL = sL;
  dutyR = sR;
  setMotor(PIN_L_PWM, PIN_L_IN1, PIN_L_IN2, dutyL);
  setMotor(PIN_R_PWM, PIN_R_IN1, PIN_R_IN2, dutyR);
}

void telemetry() {
  Bridge.notify("telemetry", (uint32_t)millis(), (int32_t)odomX, (int32_t)odomY,
                (int32_t)(odomTh * 57295.78), (int32_t)measV, (int32_t)(measW * 1000), (int32_t)ticksL,
                (int32_t)ticksR, rangeMm, readFlags(), (int32_t)batteryMv(), headDdeg);
}

void setup() {
  Bridge.begin();
  Bridge.provide_safe("cmd_vel", onCmdVel);
  Bridge.provide_safe("head", onHead);
  Bridge.provide_safe("stop", onStop);
  Bridge.provide_safe("estop", onEstop);
  Bridge.provide_safe("reset_odom", onResetOdom);

  analogReadResolution(12);
  pinMode(PIN_STBY, OUTPUT);
  digitalWrite(PIN_STBY, HIGH);
  for (int p : {PIN_L_PWM, PIN_L_IN1, PIN_L_IN2, PIN_R_PWM, PIN_R_IN1, PIN_R_IN2}) pinMode(p, OUTPUT);
  for (int p : {PIN_ENC_LA, PIN_ENC_LB, PIN_ENC_RA, PIN_ENC_RB, PIN_BUMP_L, PIN_BUMP_R, PIN_ESTOP})
    pinMode(p, INPUT_PULLUP);
  for (int i = 0; i < 4; i++) pinMode(PIN_CLIFF[i], INPUT);
  pinMode(PIN_CHARGE, INPUT);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_LA), encL, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_RA), encR, CHANGE);
  haltAll();

  Wire.begin();
  Wire.setClock(400000);
  tof.setTimeout(50);
  tofOk = tof.init();
  if (tofOk) {
    tof.setDistanceMode(VL53L1X::Short);
    tof.setMeasurementTimingBudget(20000);
    tof.startContinuous(20);
  }
  imuOk = imu.begin();
  if (imuOk) {
    imu.setGyroRange(MPU6050_RANGE_250_DEG);
    imu.setFilterBandwidth(MPU6050_BAND_44_HZ);
    float sum = 0;                                     // initial gyro bias: keep still at power-on
    for (int i = 0; i < 200; i++) {
      sensors_event_t a, g, t;
      imu.getEvent(&a, &g, &t);
      sum += g.gyro.z * 57.29578f;
      delay(5);
    }
    gyroBias = sum / 200;
  }
  head.attach(PIN_SERVO);
  head.write(90);
  lastControlUs = micros();
}

void loop() {
  if (tofOk && tof.dataReady()) {
    rangeMm = tof.read(false);
    lastRangeMs = millis();
  }
  uint32_t now = micros();
  if (now - lastControlUs >= CONTROL_US) {
    control((now - lastControlUs) / 1e6f);
    lastControlUs = now;
  }
  if (millis() - lastTelemetryMs >= TELEMETRY_MS) {
    lastTelemetryMs = millis();
    telemetry();
  }
}
