from .configuration_orbbec import OrbbecColorCameraConfig, OrbbecDepthCameraConfig

try:
    from .camera_orbbec import (
        OrbbecColorCamera,
        OrbbecDepthCamera,
        SharedOrbbecColorCamera,
        SharedOrbbecDepthCamera,
        SharedOrbbecManager,
        find_orbbec_cameras,
    )
except ImportError:
    # Allow config imports and CLI module loading on machines without the Orbbec SDK.
    pass
