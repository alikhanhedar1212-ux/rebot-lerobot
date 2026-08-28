import sys

from . import camera_realsense as _camera_realsense

sys.modules[__name__] = _camera_realsense
