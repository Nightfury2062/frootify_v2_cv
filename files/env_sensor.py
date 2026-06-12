"""
env_sensor.py — Frootify V2
Reads temperature and humidity from a DHT11 or DHT22 sensor.

Sensor wiring:
  DHT DATA pin → GPIO 4 (BCM) via a 10 kΩ pull-up resistor to 3.3 V

Sampling strategy: read once per hour (controlled by app.py).
Continuous polling is not needed — temperature/humidity change slowly.
"""

import time
import threading

# ── Thresholds ─────────────────────────────────────────────────────────────────
TEMP_THRESHOLD_C  = 30.0    # °C
HUMID_THRESHOLD   = 80.0    # %RH

# ── Sensor config ──────────────────────────────────────────────────────────────
DHT_PIN        = 4          # BCM GPIO pin
DHT_MODEL      = "DHT22"    # "DHT11" or "DHT22"
READ_INTERVAL  = 3600       # seconds between readings (1 hour)

_dht_available = False
_dht_device    = None

try:
    import adafruit_dht
    import board

    _pin_map = {
        4:  board.D4,
        17: board.D17,
        27: board.D27,
        22: board.D22,
    }
    _board_pin = _pin_map.get(DHT_PIN, board.D4)

    if DHT_MODEL == "DHT22":
        _dht_device = adafruit_dht.DHT22(_board_pin, use_pulseio=False)
    else:
        _dht_device = adafruit_dht.DHT11(_board_pin, use_pulseio=False)

    _dht_available = True
    print(f"[EnvSensor] {DHT_MODEL} on GPIO {DHT_PIN} ready.")

except (ImportError, NotImplementedError, ValueError) as e:
    print(f"[EnvSensor] adafruit-circuitpython-dht not available ({e}). Simulation mode.")


# ── Internal state (thread-safe) ───────────────────────────────────────────────
_lock             = threading.Lock()
_last_temp        = None      # °C  or None if never read
_last_humidity    = None      # %RH or None if never read
_last_read_time   = 0.0       # epoch seconds
_env_alert_active = False     # True when thresholds exceeded


# ── Sensor reading ─────────────────────────────────────────────────────────────

def _read_sensor_once():
    """
    Attempt to read the DHT sensor.
    DHT sensors occasionally return None / raise on bad pulses — retry up to 3×.
    Returns (temperature_C, humidity_pct) or (None, None) on failure.
    """
    if not _dht_available or _dht_device is None:
        # Simulation: return values that are within safe range
        import random
        sim_temp  = round(20.0 + random.uniform(-2, 12), 1)
        sim_humid = round(50.0 + random.uniform(-10, 35), 1)
        print(f"[EnvSensor SIM] Temp={sim_temp} °C  Humidity={sim_humid} %")
        return sim_temp, sim_humid

    for attempt in range(3):
        try:
            temp  = _dht_device.temperature
            humid = _dht_device.humidity
            if temp is not None and humid is not None:
                return float(temp), float(humid)
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"[EnvSensor] Read failed after 3 attempts: {e}")

    return None, None


def _evaluate_alert(temp, humid):
    """Return True if either threshold is exceeded."""
    if temp is None or humid is None:
        return False
    return temp > TEMP_THRESHOLD_C or humid > HUMID_THRESHOLD


# ── Public API ─────────────────────────────────────────────────────────────────

def read_now():
    """
    Force an immediate sensor read and update internal state.
    Call this from the hourly scheduler in app.py.
    Returns (temp, humidity, alert_active).
    """
    global _last_temp, _last_humidity, _last_read_time, _env_alert_active

    temp, humid = _read_sensor_once()

    with _lock:
        _last_temp        = temp
        _last_humidity    = humid
        _last_read_time   = time.time()
        _env_alert_active = _evaluate_alert(temp, humid)

    status = "⚠ ALERT" if _env_alert_active else "OK"
    print(
        f"[EnvSensor] Temp={temp} °C  Humidity={humid} %  → {status}"
        f"  (thresholds: >{TEMP_THRESHOLD_C} °C or >{HUMID_THRESHOLD} %)"
    )
    return temp, humid, _env_alert_active


def get_last_reading():
    """
    Return the most recent cached reading without hitting the sensor.
    Returns (temp, humidity, alert_active).  temp/humidity may be None
    if no read has been performed yet.
    """
    with _lock:
        return _last_temp, _last_humidity, _env_alert_active


def seconds_since_last_read():
    with _lock:
        return time.time() - _last_read_time


def cleanup():
    """Release the DHT device handle."""
    if _dht_available and _dht_device is not None:
        try:
            _dht_device.exit()
        except Exception:
            pass
    print("[EnvSensor] Cleaned up.")
