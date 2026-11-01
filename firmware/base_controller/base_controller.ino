// Base controller for the companion robot: wheel speed loops, reflexes and telemetry for the Pi.
//
// Board: Raspberry Pi Pico (RP2040), Arduino-Pico core (earlephilhower). Libraries: Pololu VL53L1X,
// Adafruit MPU6050 (or swap in your IMU), Servo. The Pi talks to it over USB serial (see
// robot/body/serial_body.py for the protocol). Everything safety-critical lives here, so it never
// depends on Linux scheduling:
//   * watchdog: no speed command for 300 ms -> motors stop
//   * forward stop: driving forward is refused while the range sensor reads <= 120 mm (turning is allowed)
//   * bumper / cliff / lifted / emergency stop -> motors stop until the host sends a new command away
//
// Adapt the PIN_* constants and TICKS_PER_REV to your wiring and motors. Skid steer: the two left motors
// share one driver channel, the two right motors the other; one encoder per side is enough.

#include <Wire.h>
#include <Servo.h>
#include <VL53L1X.h>
#include <Adafruit_MPU6050.h>

// ---- wiring -------------------------------------------------------------------------------------
const int PIN_L_PWM = 2, PIN_L_IN1 = 3, PIN_L_IN2 = 4;     // TB6612FNG channel A (left side)
const int PIN_R_PWM = 5, PIN_R_IN1 = 6, PIN_R_IN2 = 7;     // channel B (right side)
const int PIN_STBY = 8;
const int PIN_ENC_LA = 10, PIN_ENC_LB = 11, PIN_ENC_RA = 12, PIN_ENC_RB = 13;
const int PIN_BUMP_L = 14, PIN_BUMP_R = 15;                // microswitches to GND (active low)
const int PIN_CLIFF[4] = {16, 17, 18, 19};                 // IR floor sensors, LOW = floor seen
const int PIN_SERVO = 20;
const int PIN_ESTOP = 21;                                  // latching button to GND
const int PIN_BATT = 26;                                   // ADC0 through a divider
const float BATT_DIVIDER = 3.0;                            // Vbatt = Vadc * BATT_DIVIDER

// ---- robot --------------------------------------------------------------------------------------
const float WHEEL_DIAMETER_MM = 120.0, TRACK_MM = 208.0, TURN_SLIP = 2.4;  // as config/body.yaml
const float TICKS_PER_REV = 720.0;
const float MM_PER_TICK = 3.14159265f * WHEEL_DIAMETER_MM / TICKS_PER_REV;
const float MAX_V = 400.0, MAX_W = 120.0, ACCEL = 900.0, ANG_ACCEL = 600.0;
const float STOP_MM = 120.0;
const uint32_t WATCHDOG_MS = 300, CONTROL_US = 10000, TELEMETRY_MS = 20;
const float KP = 0.0025, KI = 0.02;                        // per side: (mm/s error) -> PWM fraction

volatile long ticksL = 0, ticksR = 0;
VL53L1X tof;
Adafruit_MPU6050 imu;
Servo head;

float targetV = 0, targetW = 0, cmdV = 0, cmdW = 0;        // set point and ramped command
float integL = 0, integR = 0, gyroZ = 0, gyroBias = 0;
long lastTicksL = 0, lastTicksR = 0;
uint32_t lastCmdMs = 0, lastControlUs = 0, lastTelemetryMs = 0;
int tofMm = 2000;
bool latched = false;                                      // a reflex stopped the base
bool estopCmd = false;

void encL() { ticksL += (digitalRead(PIN_ENC_LA) == digitalRead(PIN_ENC_LB)) ? 1 : -1; }
void encR() { ticksR += (digitalRead(PIN_ENC_RA) == digitalRead(PIN_ENC_RB)) ? -1 : 1; }

void setMotor(int pwmPin, int in1, int in2, float duty) {
  duty = constrain(duty, -1.0f, 1.0f);
  digitalWrite(in1, duty > 0);
  digitalWrite(in2, duty < 0);
  analogWrite(pwmPin, (int)(fabsf(duty) * 255));
}

void stopMotors() {
  targetV = targetW = cmdV = cmdW = 0;
  integL = integR = 0;
  setMotor(PIN_L_PWM, PIN_L_IN1, PIN_L_IN2, 0);
  setMotor(PIN_R_PWM, PIN_R_IN1, PIN_R_IN2, 0);
}

int flags() {
  int f = 0;
  if (!digitalRead(PIN_BUMP_L) || !digitalRead(PIN_BUMP_R)) f |= 1;
  int floorless = 0;
  for (int i = 0; i < 4; i++) floorless += digitalRead(PIN_CLIFF[i]) ? 1 : 0;
  if (floorless == 1 || floorless == 2) f |= 2;            // one or two edges: a drop
  if (floorless >= 3) f |= 8;                              // most sensors see nothing: picked up
  if (!digitalRead(PIN_ESTOP) || estopCmd) f |= 4;
  if (targetV > 0 && tofMm <= STOP_MM) f |= 32;
  return f;
}

void handleLine(char *line) {
  if (line[0] == 'V') {
    long v = 0, w = 0;
    if (sscanf(line + 1, "%ld %ld", &v, &w) == 2) {
      targetV = constrain((float)v, -MAX_V, MAX_V);
      targetW = constrain(w / 1000.0f, -MAX_W, MAX_W);
      lastCmdMs = millis();
      if (targetV <= 0) latched = false;                   // backing away or turning clears a latch
    }
  } else if (line[0] == 'H') {
    long d = 0;
    if (sscanf(line + 1, "%ld", &d) == 1) head.write(constrain(90 + (int)(d / 10), 65, 125));
  } else if (line[0] == 'S') {
    stopMotors();
    latched = false;
  } else if (line[0] == 'E') {
    estopCmd = line[2] == '1';
    if (estopCmd) stopMotors();
  }
}

void readSerial() {
  static char buf[48];
  static int n = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || n >= (int)sizeof(buf) - 1) {
      buf[n] = 0;
      if (n) handleLine(buf);
      n = 0;
    } else if (c != '\r') {
      buf[n++] = c;
    }
  }
}

void control(float dt) {
  sensors_event_t a, g, temp;
  imu.getEvent(&a, &g, &temp);
  gyroZ = (g.gyro.z - gyroBias) * 57.29578f;               // deg/s
  int f = flags();
  bool hard = f & (1 | 2 | 4 | 8);
  if (hard) latched = true;
  float v = targetV, w = targetW;
  if (millis() - lastCmdMs > WATCHDOG_MS || latched || (f & 4)) v = w = 0;
  if (v > 0 && tofMm <= STOP_MM) v = 0;                    // forward stop; turning is still allowed
  cmdV += constrain(v - cmdV, -ACCEL * dt, ACCEL * dt);
  cmdW += constrain(w - cmdW, -ANG_ACCEL * dt, ANG_ACCEL * dt);
  if (v == 0 && w == 0 && fabsf(cmdV) < 1 && fabsf(cmdW) < 1) {
    setMotor(PIN_L_PWM, PIN_L_IN1, PIN_L_IN2, 0);
    setMotor(PIN_R_PWM, PIN_R_IN1, PIN_R_IN2, 0);
    integL = integR = 0;
    return;
  }
  float turn = cmdW * 0.0174533f * TRACK_MM * 0.5f * TURN_SLIP;
  float wantL = cmdV - turn, wantR = cmdV + turn;          // wheel surface speeds, mm/s
  long tl = ticksL, tr = ticksR;
  float haveL = (tl - lastTicksL) * MM_PER_TICK / dt, haveR = (tr - lastTicksR) * MM_PER_TICK / dt;
  lastTicksL = tl;
  lastTicksR = tr;
  integL = constrain(integL + (wantL - haveL) * dt, -400.0f, 400.0f);
  integR = constrain(integR + (wantR - haveR) * dt, -400.0f, 400.0f);
  float ff = 1.0f / (MAX_V * 1.2f);                        // feed-forward: PWM per mm/s
  setMotor(PIN_L_PWM, PIN_L_IN1, PIN_L_IN2, wantL * ff + KP * (wantL - haveL) + KI * integL);
  setMotor(PIN_R_PWM, PIN_R_IN1, PIN_R_IN2, wantR * ff + KP * (wantR - haveR) + KI * integR);
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  pinMode(PIN_STBY, OUTPUT);
  digitalWrite(PIN_STBY, HIGH);
  for (int p : {PIN_L_PWM, PIN_L_IN1, PIN_L_IN2, PIN_R_PWM, PIN_R_IN1, PIN_R_IN2}) pinMode(p, OUTPUT);
  for (int p : {PIN_ENC_LA, PIN_ENC_LB, PIN_ENC_RA, PIN_ENC_RB, PIN_BUMP_L, PIN_BUMP_R, PIN_ESTOP})
    pinMode(p, INPUT_PULLUP);
  for (int i = 0; i < 4; i++) pinMode(PIN_CLIFF[i], INPUT);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_LA), encL, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_RA), encR, CHANGE);
  Wire.begin();
  Wire.setClock(400000);
  tof.setTimeout(50);
  tof.init();
  tof.setDistanceMode(VL53L1X::Short);
  tof.setMeasurementTimingBudget(20000);
  tof.startContinuous(20);
  imu.begin();
  imu.setGyroRange(MPU6050_RANGE_250_DEG);
  float sum = 0;                                           // gyro bias: keep the robot still at power-on
  for (int i = 0; i < 200; i++) {
    sensors_event_t a, g, t;
    imu.getEvent(&a, &g, &t);
    sum += g.gyro.z;
    delay(5);
  }
  gyroBias = sum / 200;
  head.attach(PIN_SERVO);
  head.write(90);
  stopMotors();
}

void loop() {
  readSerial();
  if (tof.dataReady()) tofMm = tof.read(false);
  uint32_t now = micros();
  if (now - lastControlUs >= CONTROL_US) {
    control((now - lastControlUs) / 1e6f);
    lastControlUs = now;
  }
  if (millis() - lastTelemetryMs >= TELEMETRY_MS) {
    lastTelemetryMs = millis();
    int mv = (int)(analogRead(PIN_BATT) * 3300.0f / 4095.0f * BATT_DIVIDER);
    Serial.printf("T %lu %ld %ld %ld %d %d %d\n", (unsigned long)millis(), ticksL, ticksR,
                  (long)(gyroZ * 1000), tofMm, flags(), mv);
  }
}
