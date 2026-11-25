#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <string_view>

constexpr int LOW = 0, HIGH = 1, OUTPUT = 2, INPUT = 3, INPUT_PULLUP = 4, CHANGE = 5;
constexpr int A0 = 14, A1 = 15, A2 = 16, A3 = 17, A4 = 18;
int pins[32]{}, pwm[32]{};
uint32_t nowMs = 0;
uint32_t millis() { return nowMs; }
uint32_t micros() { return nowMs * 1000; }
void delay(uint32_t ms) { nowMs += ms; }
void pinMode(int pin, int mode) { pins[pin] = mode == INPUT_PULLUP ? HIGH : LOW; }
void digitalWrite(int pin, int value) { pins[pin] = value; }
int digitalRead(int pin) { return pins[pin]; }
void analogWrite(int pin, int value) { pwm[pin] = value; }
int analogRead(int) { return 4095; }
void analogReadResolution(int) {}
int digitalPinToInterrupt(int pin) { return pin; }
void attachInterrupt(int, void (*)(), int) {}
using std::max, std::min;
template<class T> T constrain(T value, T lo, T hi) { return std::clamp(value, lo, hi); }

struct k_spinlock {};
int k_spin_lock(k_spinlock *) { return 0; }
void k_spin_unlock(k_spinlock *, int) {}
uint32_t k_uptime_get_32() { return nowMs; }
struct k_timer {
  void (*expire)(k_timer *);
  uint32_t next, period;
};
#define K_MSEC(ms) (ms)
#define K_TIMER_DEFINE(name, cb, stop) k_timer name{cb, 0, 0}
void k_timer_start(k_timer *timer, uint32_t delay, uint32_t period) {
  timer->next = nowMs + delay;
  timer->period = period;
}

struct {
  void begin() {}
  template<class Fn> void provide_safe(const char *, Fn) {}
  template<class... Args> void notify(const char *, Args...) {}
} Bridge;
struct {
  void begin() {}
  void setClock(int) {}
} Wire;
struct Servo {
  void attach(int pin) { pinMode(pin, OUTPUT); }
  void write(int) {}
};
struct sensors_event_t {
  struct { float z; } gyro;
  struct { float x, y, z; } acceleration;
};
constexpr int MPU6050_RANGE_250_DEG = 0, MPU6050_BAND_44_HZ = 0;
struct Adafruit_MPU6050 {
  bool begin() { return false; }
  void setGyroRange(int) {}
  void setFilterBandwidth(int) {}
  void getEvent(sensors_event_t *, sensors_event_t *, sensors_event_t *) {}
};
#define VL53L5CX_NB_TARGET_PER_ZONE 1
struct VL53L5CX_ResultsData {
  uint8_t nb_target_detected[64], target_status[64];
  int16_t distance_mm[64];
};
VL53L5CX_ResultsData frame{};
bool frameReady = false, readOk = true, readyOk = true;
void (*duringRangeRead)() = nullptr;
enum class SF_VL53L5CX_ERROR_TYPE { NO_ERROR, CANNOT_GET_DATA_READY };
struct SparkFun_VL53L5CX {
  struct {
    SF_VL53L5CX_ERROR_TYPE lastErrorCode;
  } lastError;
  bool begin() { return true; }
  void setResolution(int) {}
  void setRangingFrequency(int) {}
  void startRanging() {}
  bool isDataReady() {
    lastError.lastErrorCode = readyOk ? SF_VL53L5CX_ERROR_TYPE::NO_ERROR : SF_VL53L5CX_ERROR_TYPE::CANNOT_GET_DATA_READY;
    return readyOk && frameReady;
  }
  bool getRangingData(VL53L5CX_ResultsData *data) {
    if (duringRangeRead) duringRangeRead();
    frameReady = false;
    *data = frame;
    return readOk;
  }
};

#include "../firmware/ventuno_mcu/ventuno_mcu.ino"

void advance(uint32_t ms) {
  uint32_t end = nowMs + ms;
  while (safetyTimer.next <= end) {
    nowMs = safetyTimer.next;
    safetyTimer.next += safetyTimer.period;
    safetyTimer.expire(&safetyTimer);
  }
  nowMs = end;
}

void publishFrame(int validZones = 64, int distance = 240, int status = 5, int targets = 1) {
  for (int i = 0; i < 64; i++) {
    frame.distance_mm[i] = distance;
    frame.target_status[i] = status;
    frame.nb_target_detected[i] = i < validZones ? targets : 0;
  }
  frameReady = true;
  loop();
}

void assertDriving() {
  assert(pins[PIN_STBY] == HIGH);
  assert(pwm[PIN_L_PWM] > 0 && pwm[PIN_R_PWM] > 0);
}

void assertStopped() {
  assert(pins[PIN_STBY] == LOW);
  assert(pwm[PIN_L_PWM] == 0 && pwm[PIN_R_PWM] == 0);
  assert(cmdV == 0 && cmdW == 0);
}

int main(int argc, char **argv) {
  assert(argc == 2);
  std::string_view scenario = argv[1];
  setup();
  assertStopped();
  onCmdVel(60, 15000);
  advance(WATCHDOG_MS + 5);
  loop();
  assertStopped();
  assert(readFlags() & 256);
  publishFrame(32, 240, 9);
  onCmdVel(60, 15000);
  const auto commandTime = nowMs;
  for (int i = 0; i < 10; i++) {
    advance(5);
    loop();
  }
  assertDriving();

  if (scenario == "independent safety pins") {
    const int switches[] = {PIN_BUMP_L, PIN_BUMP_R, PIN_ESTOP, PIN_CHARGE};
    for (int pin : PIN_CLIFF) {
      for (int other : switches) assert(pin != other);
      pins[pin] = HIGH;
      advance(5);
      assert(pins[PIN_STBY] == LOW && (readFlags() & 2));
      assert(!(readFlags() & (1 | 4 | 16)));
      assert(!onResetFaults());
      pins[pin] = LOW;
      assert(onResetFaults());
      onCmdVel(60, 0);
      advance(5);
      loop();
      assertDriving();
    }
  } else if (scenario == "e-stop" || scenario == "left bumper" || scenario == "right bumper") {
    int pin = scenario == "e-stop" ? PIN_ESTOP : scenario == "left bumper" ? PIN_BUMP_L : PIN_BUMP_R;
    pins[pin] = LOW;
    advance(5);
    assert(pins[PIN_STBY] == LOW);
    assert(pwm[PIN_L_PWM] > 0 && pwm[PIN_R_PWM] > 0);
    loop();
    assertStopped();
    assert(!onResetFaults());
    pins[pin] = HIGH;
    onCmdVel(0, 0);
    onStop();
    onCmdVel(0, 15000);
    advance(5);
    loop();
    assertStopped();
    assert(onResetFaults());
    advance(5);
    loop();
    assertStopped();
    onCmdVel(60, 0);
    advance(5);
    loop();
    assertDriving();
  } else if (scenario == "late range frame") {
    while (nowMs - lastRangeMs < 240) {
      advance(5);
      ticksL++;
      ticksR++;
      loop();
    }
    assertDriving();
    duringRangeRead = [] {
      advance(9);
      assertDriving();
      advance(1);
      assert(pins[PIN_STBY] == LOW);
      assert(pwm[PIN_L_PWM] > 0 && pwm[PIN_R_PWM] > 0);
    };
    publishFrame();
    duringRangeRead = nullptr;
    assertStopped();
    assert(readFlags() & 256);
    assert(!(readFlags() & 2048));
    onCmdVel(60, 0);
    advance(5);
    loop();
    assertStopped();
    assert(onResetFaults());
    onCmdVel(60, 0);
    advance(5);
    loop();
    assertDriving();
  } else if (scenario == "command timeout" || scenario == "stalled control") {
    if (scenario == "command timeout") {
      while (nowMs - commandTime < 295) {
        advance(5);
        ticksL++;
        ticksR++;
        publishFrame();
      }
      assertDriving();
      advance(4);
      assertDriving();
      advance(1);
    } else {
      onCmdVel(60, 0);
      advance(49);
      assertDriving();
      advance(1);
    }
    // No call to loop() or control() at the cutoff: only STBY changes.
    assert(pins[PIN_STBY] == LOW);
    assert(pwm[PIN_L_PWM] > 0 && pwm[PIN_R_PWM] > 0);
    assert(readFlags() & 2048);
    onCmdVel(60, 0);
    loop();
    assertStopped();
  } else {
    advance(scenario == "readiness failure" ? 1 : 5);
    const auto lastGoodFrame = lastRangeMs;
    if (scenario == "invalid distance") publishFrame(64, 0);
    else if (scenario == "invalid status") publishFrame(64, 240, 255);
    else if (scenario == "missing targets") publishFrame(64, 240, 5, 0);
    else if (scenario == "insufficient coverage") publishFrame(31);
    else if (scenario == "read failure") {
      readOk = false;
      publishFrame();
      readOk = true;
    } else if (scenario == "readiness failure") {
      loop();
      assertDriving();
      readyOk = false;
      loop();
      readyOk = true;
    } else {
      assert(false);
    }
    assertStopped();
    assert(readFlags() & 256);
    assert(minRangeMm == -1 && lastRangeMs == lastGoodFrame);
    onCmdVel(-60, 15000);
    advance(5);
    loop();
    assertStopped();
    publishFrame();
    advance(5);
    loop();
    assertStopped();
    onCmdVel(60, 0);
    advance(5);
    loop();
    assertDriving();
  }
}
