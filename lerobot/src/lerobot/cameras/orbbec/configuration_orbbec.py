from dataclasses import dataclass
from typing import Optional

from ..configs import CameraConfig, ColorMode, Cv2Rotation


def _validate_rotation(rotation: Cv2Rotation) -> None:
    if rotation not in (
        Cv2Rotation.NO_ROTATION,
        Cv2Rotation.ROTATE_90,
        Cv2Rotation.ROTATE_180,
        Cv2Rotation.ROTATE_270,
    ):
        raise ValueError(
            f"`rotation` is expected to be in "
            f"{(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, "
            f"but {rotation} is provided."
        )


def _validate_stream_shape(fps: int | None, width: int | None, height: int | None) -> None:
    values = (fps, width, height)
    if any(v is not None for v in values) and any(v is None for v in values):
        raise ValueError("For `fps`, `width` and `height`, either all of them need to be set, or none of them.")


def _validate_positive_float(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"`{name}` is expected to be > 0, but {value} is provided.")


@CameraConfig.register_subclass("orbbec_color")
@dataclass(kw_only=True)
class OrbbecColorCameraConfig(CameraConfig):
    serial_number_or_name: str
    fps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    def __post_init__(self):
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, "
                f"but {self.color_mode} is provided."
            )

        _validate_rotation(self.rotation)
        _validate_stream_shape(self.fps, self.width, self.height)


@CameraConfig.register_subclass("orbbec_depth")
@dataclass(kw_only=True)
class OrbbecDepthCameraConfig(CameraConfig):
    serial_number_or_name: str
    fps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

    depth_alpha: float = 0.2
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 5

    def __post_init__(self):
        _validate_rotation(self.rotation)
        _validate_stream_shape(self.fps, self.width, self.height)
        _validate_positive_float("depth_alpha", self.depth_alpha)
