"""
gpio_manager.py — Frootify V2
Handles both LEDs via RPi.GPIO.

Pin assignments (BCM numbering):
  GPIO 17  →  Red LED   (Rotten tomato warning)
  GPIO 27  →  Yellow LED (Environmental warning: temp / humidity)

Wiring reminder:
  Pi GPIO pin → 330 Ω resistor → LED anode → LED cathode → GND
"""

import time

# ── Pin numbers (BCM) ──────────────────────────────────────────────────────────
PIN_LED_ROTTEN = 17      # Red   LED
PIN_LED_ENV    = 27      # Yellow/Green LED

_gpio_available = False

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(PIN_LED_ROTTEN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PIN_LED_ENV,    GPIO.OUT, initial=GPIO.LOW)
    _gpio_available = True
    print("[GPIO] RPi.GPIO initialised successfully.")
except (ImportError, RuntimeError) as e:
    print(f"[GPIO] RPi.GPIO not available ({e}). Running in simulation mode.")


# ── Public API ─────────────────────────────────────────────────────────────────

def set_rotten_led(state: bool):
    """Turn the rotten-tomato warning LED on (True) or off (False)."""
    if _gpio_available:
        GPIO.output(PIN_LED_ROTTEN, GPIO.HIGH if state else GPIO.LOW)
    else:
        print(f"[GPIO SIM] Rotten LED → {'ON ' if state else 'OFF'}")


def set_env_led(state: bool):
    """Turn the environmental warning LED on (True) or off (False)."""
    if _gpio_available:
        GPIO.output(PIN_LED_ENV, GPIO.HIGH if state else GPIO.LOW)
    else:
        print(f"[GPIO SIM] Env    LED → {'ON ' if state else 'OFF'}")


def all_off():
    """Turn both LEDs off."""
    set_rotten_led(False)
    set_env_led(False)


def test_leds(duration: float = 0.5):
    """
    Quick startup self-test: blink both LEDs once so you can confirm
    the wiring is correct before the main loop begins.
    """
    print("[GPIO] Running LED self-test …")
    set_rotten_led(True)
    time.sleep(duration)
    set_rotten_led(False)

    set_env_led(True)
    time.sleep(duration)
    set_env_led(False)

    print("[GPIO] LED self-test complete.")


def cleanup():
    """Release GPIO resources. Call on shutdown."""
    all_off()
    if _gpio_available:
        GPIO.cleanup()
    print("[GPIO] Cleaned up.")
