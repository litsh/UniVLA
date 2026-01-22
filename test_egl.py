import mujoco
import os
# Ensure the env var is set in the current process
os.environ["MUJOCO_GL"] = "egl"

import OpenGL
OpenGL.ERROR_CHECKING = False

try:
    # This will attempt to initialize an EGL context
    ctx = mujoco.GLContext(640, 480)
    ctx.make_current()
    print("Success: MuJoCo is using GPU rendering (EGL)!")
    ctx.free()
except Exception as e:
    print(f"Failed to initialize GPU rendering: {e}")