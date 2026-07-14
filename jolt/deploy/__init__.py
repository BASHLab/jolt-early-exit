"""Device-side deployment seam.

The real device-side measurement code (ESP32-S3 quantization via ESP-PPQ/ESP-DL, energy
capture with Keysight EDU36311A/EDU34450A, Raspberry Pi 5 latency) lives in a separate repo
and plugs in behind the transport interface defined in ``split_inference``. This package
ships only the clean seam, not the measurement code.
"""
