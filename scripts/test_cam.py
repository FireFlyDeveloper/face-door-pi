#!/usr/bin/env python3
"""Quick camera test."""
from picamera2 import Picamera2
import time

cam = Picamera2()
config = cam.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
cam.configure(config)
cam.start()
time.sleep(2)
frame = cam.capture_array()
print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
cam.stop()
cam.close()
print("Camera OK")
