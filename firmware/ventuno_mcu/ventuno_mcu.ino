// Real-time base controller for the Arduino VENTUNO Q's STM32H5 (Cortex-M33, Arduino core on Zephyr).
//
// The Qualcomm side (Linux) thinks and plans; this MCU owns everything that must happen on time:
//   * 200 Hz wheel velocity loops: feed-forward + PI with anti-windup, filtered encoder speed, accel limits
//   * Odometry at 200 Hz: distance from encoders, heading from gyro (skid-steer slip compensation);
//     gyro bias is re-learned whenever the robot stands still
//   * VL53L5CX 8x8 (64-zone) ToF step-vs-cliff reflex:
//     - |dz| <= 22 mm: traversable (with 1.35x torque-boost for step-up dz >= 8 mm)
//     - dz < -40 mm: drop cliff stop (instant halt, latches motors off)
//     - dz > 40 mm: obstacle
//   * Sub-150 ms anti-cable / wheel snag reflex:
//     - Cuts forward drive within 150 ms if cmdV / duty is high while wheel speed stalls on flat pitch
//   * Interrupt-timed motor gate: command loss or stalled control -> standby
//   * 50 Hz telemetry to Linux: pose, velocities, raw ticks, range, fault flags, battery, head
//
// Link: Arduino Router Bridge (msgpack-RPC; Linux side is robot/hardware/ventuno_body.py over
// /var/run/arduino-router.sock). Methods:
//   cmd_vel(v_mm_s, w_mdeg_s)   head(tilt_ddeg)   stop()   estop(on)   reset_faults()   reset_odom()
// Telemetry notification:
//   [ms, x_mm, y_mm, heading_mdeg, v_mm_s, w_mdeg_s, ticks_l, ticks_r, range_mm, flags, battery_mv, head_ddeg]
// flags: 1 bumper, 2 cliff, 4 estop, 8 picked up, 16 charging, 32 forward blocked, 64 wheel stall/snag,
//        128 IMU fault, 256 range sensor fault, 512 battery low, 1024 torque boost active, 2048 watchdog

#include <Arduino_RouterBridge.h>
#include <Wire.h>
#include <Servo.h>
#include <SparkFun_VL53L5CX_Library.h>
#include <Adafruit_MPU6050.h>
#include <zephyr/kernel.h>

// ---- wiring --------------------------------------------------------------------------------------
const int PIN_L_PWM = 3, PIN_L_IN1 = 4, PIN_L_IN2 = 5;
const int PIN_R_PWM = 6, PIN_R_IN1 = 7, PIN_R_IN2 = 8;
const int PIN_STBY = 9;                               // active high; requires a hardware pull-down during reset
const int PIN_ENC_LA = 2, PIN_ENC_LB = 10, PIN_ENC_RA = 11, PIN_ENC_RB = 12;
const int PIN_BUMP_L = A0, PIN_BUMP_R = A1;            // microswitches to GND (active low)
const int PIN_CLIFF[4] = {25, 26, 13, 27};             // LOW = floor seen; D25–D27 are JMISC pins 1–3
const int PIN_SERVO = 28;                             // JMISC pin 4
const int PIN_ESTOP = 16;                              // latching button to GND
const int PIN_BATT = A4;                               // battery divider
const int PIN_CHARGE = 17;                             // dock contacts sense, HIGH = charging
const float BATT_DIVIDER = 3.0f, ADC_VREF = 3.3f, ADC_MAX = 4095.0f;

// ---- robot dimensions & dynamics -----------------------------------------------------------------
const float WHEEL_DIAMETER_MM = 120.0f, TRACK_MM = 208.0f, TURN_SLIP = 2.4f;
const float TICKS_PER_REV = 720.0f;
const float MM_PER_TICK = 3.14159265f * WHEEL_DIAMETER_MM / TICKS_PER_REV;
const float MAX_V = 450.0f, MAX_W = 150.0f;            // mm/s, deg/s
const float ACCEL = 1000.0f, DECEL = 1400.0f, ANG_ACCEL = 700.0f;
const float STOP_MM = 120.0f;                          // range reading at which forward motion must be 0
const float BRAKE_DECEL = 800.0f;                      // mm/s^2 safety envelope
const float BATT_LOW_MV = 6600.0f;

// Nominal expected floor distance for angled front ToF (mm)
const float EXPECTED_FLOOR_MM = 240.0f;
const float TRAVERSABLE_DZ_MM = 22.0f;
const float TORQUE_BOOST_DZ_MM = 8.0f;
const float CLIFF_DROP_DZ_MM = -40.0f;
const float OBSTACLE_DZ_MM = 40.0f;

// ---- timing --------------------------------------------------------------------------------------
const uint32_t CONTROL_US = 5000;                      // 200 Hz
const uint32_t TELEMETRY_MS = 20;                      // 50 Hz
const uint32_t WATCHDOG_MS = 300;
const uint32_t CONTROL_WATCHDOG_MS = 50;              // provisional until measured with I2C/Router loaded
const uint32_t SAFETY_POLL_MS = 5;
const uint32_t RANGE_TIMEOUT_MS = 250;
const uint32_t SNAG_TIMEOUT_MS = 150;                  // 150 ms anti-cable / wheel snag reflex
const int MIN_VALID_ZONES = 32;                        // require at least half the grid; not a floor calibration

// ---- loop gains (per side, mm/s error -> PWM fraction) --------------------------------------------
const float KFF = 1.0f / (MAX_V * 1.2f), KP = 0.0028f, KI = 0.022f, KAW = 2.0f;
const float SPEED_LPF = 0.35f;

volatile int32_t ticksL = 0, ticksR = 0;
SparkFun_VL53L5CX tof;
VL53L5CX_ResultsData tofData;
Adafruit_MPU6050 imu;
Servo head;
bool imuOk = false, tofOk = false;

// commands
float targetV = 0, targetW = 0;
uint32_t lastCmdMs = 0;
int headDdeg = 0;

// The timer ISR and sketch share the motor gate, deadlines and hard-stop latch under this lock.
struct k_spinlock safetyLock;
bool estopCmd = false, motorsEnabled = false;
uint32_t lastDriveMs = 0;
int latchedFlags = 0;

// state
float cmdV = 0, cmdW = 0;
float speedL = 0, speedR = 0, integL = 0, integR = 0;
float dutyL = 0, dutyR = 0;
int32_t lastTicksL = 0, lastTicksR = 0, odomTicksL = 0, odomTicksR = 0;
double odomX = 0, odomY = 0, odomTh = 0;               // mm, mm, rad
float gyroZ = 0, gyroBias = 0;                         // deg/s
float pitchDeg = 0;
float measV = 0, measW = 0;
uint32_t stillSinceMs = 0, snagSinceMs = 0, lastRangeMs = 0;
float biasAcc = 0;
int biasN = 0;
int minRangeMm = -1;
bool rangeValid = false, cliffDetected = false;
bool torqueBoostActive = false;
uint32_t lastControlUs = 0, lastTelemetryMs = 0;
int faults = 0;

void encL() { ticksL += (digitalRead(PIN_ENC_LA) == digitalRead(PIN_ENC_LB)) ? 1 : -1; }
void encR() { ticksR += (digitalRead(PIN_ENC_RA) == digitalRead(PIN_ENC_RB)) ? -1 : 1; }

int physicalFlags() {
  int f = 0;
  if (!digitalRead(PIN_BUMP_L) || !digitalRead(PIN_BUMP_R)) f |= 1;
  int floorless = 0;
  for (int p : PIN_CLIFF) floorless += digitalRead(p) ? 1 : 0;
  if (floorless >= 1) f |= 2;
  if (floorless >= 3) f |= 8;
  if (!digitalRead(PIN_ESTOP)) f |= 4;
  return f;
}

void setMotorGate(bool enable) {
  digitalWrite(PIN_STBY, enable ? HIGH : LOW);
  motorsEnabled = enable;
}

bool rangeFresh(uint32_t now) {
  return rangeValid && now - lastRangeMs < RANGE_TIMEOUT_MS;
}

void checkSafety(uint32_t now) {
  latchedFlags |= physicalFlags();
  if (estopCmd) latchedFlags |= 4;
  if (motorsEnabled) {
    if (!rangeFresh(now)) latchedFlags |= 256;
    if (now - lastCmdMs >= WATCHDOG_MS || now - lastDriveMs >= CONTROL_WATCHDOG_MS) latchedFlags |= 2048;
    if (latchedFlags) setMotorGate(false);
  }
}

void safetyTick(struct k_timer *) {
  auto key = k_spin_lock(&safetyLock);
  checkSafety(k_uptime_get_32());
  k_spin_unlock(&safetyLock, key);
}
K_TIMER_DEFINE(safetyTimer, safetyTick, nullptr);

bool motionBlocked(uint32_t now) {
  checkSafety(now);
  return latchedFlags || !rangeFresh(now) || now - lastCmdMs >= WATCHDOG_MS;
}

void setMotor(int pwmPin, int in1, int in2, float duty) {
  duty = constrain(duty, -1.0f, 1.0f);
  digitalWrite(in1, duty > 0);
  digitalWrite(in2, duty < 0);
  analogWrite(pwmPin, (int)(fabsf(duty) * 255));
}

void motorsOff() {
  auto key = k_spin_lock(&safetyLock);
  setMotorGate(false);
  k_spin_unlock(&safetyLock, key);
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
  auto key = k_spin_lock(&safetyLock);
  uint32_t now = k_uptime_get_32();
  checkSafety(now);
  if (latchedFlags || !rangeFresh(now)) {
    k_spin_unlock(&safetyLock, key);
    return;
  }
  lastCmdMs = now;
  k_spin_unlock(&safetyLock, key);
  targetV = constrain((float)v, -MAX_V, MAX_V);
  targetW = constrain(wMdeg / 1000.0f, -MAX_W, MAX_W);
}
void onHead(int ddeg) {
  headDdeg = constrain(ddeg, -250, 350);
  head.write(constrain(90 + headDdeg / 10, 60, 130));
}
void onStop() { haltAll(); }
void onEstop(bool on) {
  auto key = k_spin_lock(&safetyLock);
  estopCmd = on;
  checkSafety(k_uptime_get_32());
  k_spin_unlock(&safetyLock, key);
  if (on) haltAll();
}
bool onResetFaults() {
  haltAll();
  auto key = k_spin_lock(&safetyLock);
  bool healthy = !estopCmd && !physicalFlags() && rangeFresh(k_uptime_get_32()) && !cliffDetected;
  if (healthy) latchedFlags = 0;
  k_spin_unlock(&safetyLock, key);
  return healthy;
}
void onResetOdom() { odomX = odomY = odomTh = 0; }

// ---- sensing ----------------------------------------------------------------------------------------
int readFlags() {
  auto key = k_spin_lock(&safetyLock);
  int f = physicalFlags() | latchedFlags;
  k_spin_unlock(&safetyLock, key);
  if (cliffDetected) f |= 2;
  if (!rangeFresh(k_uptime_get_32())) f |= 256;
  if (digitalRead(PIN_CHARGE)) f |= 16;
  if (targetV > 0 && rangeValid && minRangeMm <= STOP_MM) f |= 32;
  if (torqueBoostActive) f |= 1024;
  return f | faults;
}

float batteryMv() { return analogRead(PIN_BATT) * ADC_VREF / ADC_MAX * BATT_DIVIDER * 1000.0f; }

// ---- control ----------------------------------------------------------------------------------------
void control(float dt) {
  uint32_t nowMs = k_uptime_get_32();
  if (imuOk) {
    sensors_event_t a, g, t;
    imu.getEvent(&a, &g, &t);
    gyroZ = g.gyro.z * 57.29578f - gyroBias;
    // Calculate pitch from accelerometer
    pitchDeg = atan2f(-a.acceleration.x, sqrtf(a.acceleration.y * a.acceleration.y + a.acceleration.z * a.acceleration.z)) * 57.29578f;
  }

  // Wheel speeds from encoders, low-passed
  int32_t tl = ticksL, tr = ticksR;
  float rawL = (tl - lastTicksL) * MM_PER_TICK / dt, rawR = (tr - lastTicksR) * MM_PER_TICK / dt;
  lastTicksL = tl;
  lastTicksR = tr;
  speedL += SPEED_LPF * (rawL - speedL);
  speedR += SPEED_LPF * (rawR - speedR);

  // Odometry: distance from wheels, heading from gyro
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

  // Re-learn gyro bias while stationary
  bool still = fabsf(speedL) < 2 && fabsf(speedR) < 2 && targetV == 0 && targetW == 0;
  if (!still) { stillSinceMs = nowMs; biasAcc = 0; biasN = 0; }
  else if (imuOk && nowMs - stillSinceMs > 1000) {
    biasAcc += gyroZ;
    if (++biasN >= 200) { gyroBias += biasAcc / biasN; biasAcc = 0; biasN = 0; }
  }

  // Faults and reflexes
  faults = 0;
  if (!imuOk) faults |= 128;
  if (batteryMv() < BATT_LOW_MV) faults |= 512;

  // Sub-150 ms anti-cable / wheel snag reflex on flat pitch (< 7 deg):
  // High drive command while actual wheel speed stalls
  bool highCmd = (fabsf(targetV) > 80.0f) || (fabsf(dutyL) > 0.4f || fabsf(dutyR) > 0.4f);
  bool wheelStalled = fabsf(speedL) < 15.0f && fabsf(speedR) < 15.0f;
  bool flatPitch = fabsf(pitchDeg) < 7.0f;

  if (highCmd && wheelStalled && flatPitch) {
    if (nowMs - snagSinceMs >= SNAG_TIMEOUT_MS) {
      faults |= 64;  // wheel snag / stall latch
    }
  } else {
    snagSinceMs = nowMs;
  }

  auto key = k_spin_lock(&safetyLock);
  latchedFlags |= (cliffDetected ? 2 : 0) | (faults & 64);
  bool blocked = motionBlocked(k_uptime_get_32());
  k_spin_unlock(&safetyLock, key);
  if (blocked) {
    haltAll();
    return;
  }

  // Safety envelope based on min range
  float v = targetV, w = targetW;
  float clear = max(0.0f, (float)minRangeMm - STOP_MM);
  if (v > 0) v = min(v, sqrtf(2.0f * BRAKE_DECEL * clear));

  float accV = (fabsf(v) < fabsf(cmdV) || v * cmdV < 0) ? DECEL : ACCEL;
  cmdV += constrain(v - cmdV, -accV * dt, accV * dt);
  cmdW += constrain(w - cmdW, -ANG_ACCEL * dt, ANG_ACCEL * dt);

  if (v == 0 && w == 0 && fabsf(cmdV) < 1 && fabsf(cmdW) < 1) {
    cmdV = cmdW = 0;
    motorsOff();
    return;
  }

  // Apply torque boost (1.35x) if climbing traversable step-up
  float torqueScale = torqueBoostActive ? 1.35f : 1.0f;

  float turn = cmdW * 0.0174533f * TRACK_MM * 0.5f * TURN_SLIP;
  float wantL = (cmdV - turn) * torqueScale, wantR = (cmdV + turn) * torqueScale;
  float uL = wantL * KFF + KP * (wantL - speedL) + KI * integL;
  float uR = wantR * KFF + KP * (wantR - speedR) + KI * integR;
  float sL = constrain(uL, -1.0f, 1.0f), sR = constrain(uR, -1.0f, 1.0f);

  integL += ((wantL - speedL) + KAW * (sL - uL) / KI) * dt;
  integR += ((wantR - speedR) + KAW * (sR - uR) / KI) * dt;
  dutyL = sL;
  dutyR = sR;
  setMotor(PIN_L_PWM, PIN_L_IN1, PIN_L_IN2, dutyL);
  setMotor(PIN_R_PWM, PIN_R_IN1, PIN_R_IN2, dutyR);

  key = k_spin_lock(&safetyLock);
  uint32_t now = k_uptime_get_32();
  if (now - nowMs >= CONTROL_WATCHDOG_MS) latchedFlags |= 2048;
  blocked = motionBlocked(now);
  if (!blocked) {
    lastDriveMs = now;
    if (!motorsEnabled) setMotorGate(true);
  }
  k_spin_unlock(&safetyLock, key);
  if (blocked) haltAll();
}

void telemetry() {
  Bridge.notify("telemetry", (uint32_t)millis(), (int32_t)odomX, (int32_t)odomY,
                (int32_t)(odomTh * 57295.78), (int32_t)measV, (int32_t)(measW * 1000), (int32_t)ticksL,
                (int32_t)ticksR, rangeFresh(k_uptime_get_32()) ? minRangeMm : -1, readFlags(),
                (int32_t)batteryMv(), headDdeg);
}

void setup() {
  pinMode(PIN_STBY, OUTPUT);
  setMotorGate(false);
  analogReadResolution(12);
  for (int p : {PIN_L_PWM, PIN_L_IN1, PIN_L_IN2, PIN_R_PWM, PIN_R_IN1, PIN_R_IN2}) pinMode(p, OUTPUT);
  for (int p : {PIN_ENC_LA, PIN_ENC_LB, PIN_ENC_RA, PIN_ENC_RB, PIN_BUMP_L, PIN_BUMP_R, PIN_ESTOP})
    pinMode(p, INPUT_PULLUP);
  for (int i = 0; i < 4; i++) pinMode(PIN_CLIFF[i], INPUT);
  pinMode(PIN_CHARGE, INPUT);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_LA), encL, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_RA), encR, CHANGE);
  haltAll();
  k_timer_start(&safetyTimer, K_MSEC(SAFETY_POLL_MS), K_MSEC(SAFETY_POLL_MS));

  Bridge.begin();
  Bridge.provide_safe("cmd_vel", onCmdVel);
  Bridge.provide_safe("head", onHead);
  Bridge.provide_safe("stop", onStop);
  Bridge.provide_safe("estop", onEstop);
  Bridge.provide_safe("reset_faults", onResetFaults);
  Bridge.provide_safe("reset_odom", onResetOdom);

  Wire.begin();
  Wire.setClock(1000000);  // 1 MHz Fast-Mode Plus for VL53L5CX 8x8 streaming

  // Initialize VL53L5CX 8x8 multi-zone ToF
  tofOk = tof.begin();
  if (tofOk) {
    tof.setResolution(8 * 8);             // 64 zones
    tof.setRangingFrequency(15);          // 15 Hz 8x8 frames
    tof.startRanging();
  }

  imuOk = imu.begin();
  if (imuOk) {
    imu.setGyroRange(MPU6050_RANGE_250_DEG);
    imu.setFilterBandwidth(MPU6050_BAND_44_HZ);
    float sum = 0;
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
  snagSinceMs = k_uptime_get_32();
}

void readRange() {
  if (!tofOk) return;
  bool ready = tof.isDataReady();
  if (!ready && tof.lastError.lastErrorCode == SF_VL53L5CX_ERROR_TYPE::NO_ERROR) return;
  int closest = 4000, validZones = 0;
  bool cliff = false, stepUp = false;
  if (ready && tof.getRangingData(&tofData)) {
    for (int i = 0; i < 64; i++) {
      if (!tofData.nb_target_detected[i]) continue;
      int zone = i * VL53L5CX_NB_TARGET_PER_ZONE;
      int status = tofData.target_status[zone], d = tofData.distance_mm[zone];
      if ((status != 5 && status != 9) || d <= 20 || d >= 4000) continue;
      validZones++;
      closest = min(closest, d);
      float deltaZ = EXPECTED_FLOOR_MM - (float)d;
      cliff |= deltaZ < CLIFF_DROP_DZ_MM;
      stepUp |= deltaZ >= TORQUE_BOOST_DZ_MM && deltaZ <= TRAVERSABLE_DZ_MM;
    }
  }
  auto key = k_spin_lock(&safetyLock);
  uint32_t now = k_uptime_get_32();
  if (cliff) latchedFlags |= 2;
  checkSafety(now);
  bool expired = rangeValid && !rangeFresh(now);
  rangeValid = validZones >= MIN_VALID_ZONES;
  if (rangeValid) lastRangeMs = now;
  k_spin_unlock(&safetyLock, key);

  minRangeMm = rangeValid ? closest : -1;
  cliffDetected = cliff;
  torqueBoostActive = rangeValid && stepUp;
  if (expired || !rangeValid || cliff) haltAll();
}

void loop() {
  readRange();
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
