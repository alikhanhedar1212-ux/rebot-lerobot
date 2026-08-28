# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Provides the RealSense camera integrations used by LeRobot.

Common RealSense stream profiles seen in practice on this project.
These are reference comments only. They are not enforced in code.

- D405 color: 640x480@30 rgb8 / bgr8 / yuyv
- D405 depth: 640x480@30 z16
- D435i color: 640x480@30 rgb8 / bgr8 / yuyv
- D435i color: 1280x720@30 rgb8 / bgr8 / yuyv
- D435i color: 1920x1080@30 rgb8 / bgr8 / yuyv
- D435i depth: 640x480@30 z16

If a requested stream profile is not supported, the hardware or SDK should reject
it during `pipeline.start()`. In that case, run `lerobot-find-cameras realsense`
to inspect the connected model, serial number, USB link, and default profiles.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Condition, Event, Lock, Thread
from typing import Any, Literal

import cv2  # type: ignore  # TODO: add type stubs for OpenCV
import numpy as np  # type: ignore  # TODO: add type stubs for numpy
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

try:
    import pyrealsense2 as rs  # type: ignore  # TODO: add type stubs for pyrealsense2
except Exception as e:
    rs = None
    logging.info(f"Could not import realsense: {e}")

from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_realsense import (
    RealSenseD405ColorCameraConfig,
    RealSenseD405DepthCameraConfig,
    RealSenseD435iColorCameraConfig,
    RealSenseD435iDepthCameraConfig,
)

logger = logging.getLogger(__name__)

StreamKind = Literal["color", "depth"]


@dataclass(frozen=True)
class RealSenseModelTraits:
    key: str
    model_name: str
    supported_tokens: tuple[str, ...]
    supports_max_depth_m: bool = False


D405_MODEL_TRAITS = RealSenseModelTraits(
    key="d405",
    model_name="D405",
    supported_tokens=("D405",),
)

D435I_MODEL_TRAITS = RealSenseModelTraits(
    key="d435i",
    model_name="D435i",
    supported_tokens=("D435I",),
    supports_max_depth_m=True,
)

REALSENSE_MODEL_REGISTRY: dict[str, RealSenseModelTraits] = {
    traits.key: traits for traits in (D405_MODEL_TRAITS, D435I_MODEL_TRAITS)
}

_SUPPORTED_REALSENSE_TRAITS = tuple(REALSENSE_MODEL_REGISTRY.values())


def _require_realsense() -> Any:
    if rs is None:
        raise ImportError("pyrealsense2 is required for RealSense cameras.")
    return rs


def _get_camera_info(device: Any, field: Any) -> str | None:
    try:
        value = device.get_info(field)
    except Exception:
        return None
    return str(value)


def _normalize_device_name(name: str) -> str:
    return name.upper().replace(" ", "")


def _detect_model_traits(device_info: dict[str, Any]) -> RealSenseModelTraits | None:
    normalized_name = _normalize_device_name(device_info["name"])
    for traits in REALSENSE_MODEL_REGISTRY.values():
        if any(token in normalized_name for token in traits.supported_tokens):
            return traits
    return None


def _matches_supported_model(
    device_info: dict[str, Any], supported_traits: tuple[RealSenseModelTraits, ...] | None
) -> bool:
    if supported_traits is None:
        return True
    detected_traits = _detect_model_traits(device_info)
    return detected_traits is not None and any(detected_traits.key == traits.key for traits in supported_traits)


def _detect_model_name(device_info: dict[str, str]) -> str:
    traits = _detect_model_traits(device_info)
    if traits is not None:
        return traits.model_name
    return device_info["name"]


def _make_usb_warning(usb_type_descriptor: str | None) -> str | None:
    if usb_type_descriptor and usb_type_descriptor.startswith("2"):
        return (
            f"Detected USB {usb_type_descriptor}. Color/depth streaming may stall. "
            "A USB 3.x port/cable is strongly recommended."
        )
    return None


def _extract_default_stream_profile(device: Any) -> dict[str, Any] | None:
    for sensor in device.query_sensors():
        try:
            profiles = sensor.get_stream_profiles()
        except Exception:
            continue

        for profile in profiles:
            try:
                is_default = profile.is_default()
                is_video = profile.is_video_stream_profile()
            except Exception:
                continue

            if not is_default or not is_video:
                continue

            vprofile = profile.as_video_stream_profile()
            return {
                "stream_type": vprofile.stream_name(),
                "format": vprofile.format().name,
                "width": vprofile.width(),
                "height": vprofile.height(),
                "fps": vprofile.fps(),
            }

    return None


def _query_realsense_devices() -> list[dict[str, Any]]:
    rs_module = _require_realsense()
    devices_info: list[dict[str, Any]] = []
    context = rs_module.context()
    for device in context.query_devices():
        usb_type_descriptor = _get_camera_info(device, rs_module.camera_info.usb_type_descriptor) or ""
        device_info: dict[str, Any] = {
            "name": _get_camera_info(device, rs_module.camera_info.name) or "",
            "model": "",
            "serial_number": _get_camera_info(device, rs_module.camera_info.serial_number) or "",
            "firmware_version": _get_camera_info(device, rs_module.camera_info.firmware_version) or "",
            "usb_type_descriptor": usb_type_descriptor,
            "physical_port": _get_camera_info(device, rs_module.camera_info.physical_port) or "",
            "product_line": _get_camera_info(device, rs_module.camera_info.product_line) or "",
            "product_id": _get_camera_info(device, rs_module.camera_info.product_id) or "",
            "default_stream_profile": _extract_default_stream_profile(device),
            "usb_warning": _make_usb_warning(usb_type_descriptor),
        }
        device_info["model"] = _detect_model_name(device_info)
        devices_info.append(device_info)
    return devices_info


def find_realsense_cameras() -> list[dict[str, Any]]:
    return [
        {
            "name": device["name"],
            "model": device["model"],
            "type": "RealSense",
            "id": device["serial_number"],
            "serial_number": device["serial_number"],
            "firmware_version": device["firmware_version"],
            "usb_type_descriptor": device["usb_type_descriptor"],
            "usb_warning": device["usb_warning"],
            "physical_port": device["physical_port"],
            "product_id": device["product_id"],
            "product_line": device["product_line"],
            "default_stream_profile": device["default_stream_profile"],
        }
        for device in _query_realsense_devices()
        if _matches_supported_model(device, _SUPPORTED_REALSENSE_TRAITS)
    ]


def _find_realsense_device_info(
    serial_number_or_name: str,
    supported_traits: tuple[RealSenseModelTraits, ...] | None = None,
) -> dict[str, Any]:
    devices = _query_realsense_devices()
    matches = [device for device in devices if device["serial_number"] == serial_number_or_name]
    if not matches:
        matches = [device for device in devices if device["name"] == serial_number_or_name]

    if not matches:
        available = [f'{device["model"]} {device["serial_number"]}' for device in devices]
        raise ValueError(
            f"No supported RealSense camera found for '{serial_number_or_name}'. Available devices: {available}"
        )

    if len(matches) > 1:
        serials = [device["serial_number"] for device in matches]
        raise ValueError(
            f"Multiple RealSense cameras found with name '{serial_number_or_name}'. "
            f"Please use the serial number instead. Found serials: {serials}"
        )

    device_info = matches[0]
    if not _matches_supported_model(device_info, supported_traits):
        expected = ", ".join(traits.model_name for traits in (supported_traits or ()))
        raise ValueError(
            f"Requested RealSense camera '{serial_number_or_name}' resolved to "
            f"{device_info['model']} ({device_info['serial_number']}), but this camera class "
            f"only supports: {expected}."
        )

    return device_info


def _resolve_serial_number(
    serial_number_or_name: str,
    supported_traits: tuple[RealSenseModelTraits, ...] | None = None,
) -> tuple[str, str | None]:
    if serial_number_or_name.isdigit():
        return serial_number_or_name, None

    device_info = _find_realsense_device_info(serial_number_or_name, supported_traits=supported_traits)
    return device_info["serial_number"], device_info["name"]


def _get_rs_color_format(rs_module: Any, format_name: str | None) -> Any:
    normalized_format = format_name or "rgb8"
    try:
        return getattr(rs_module.format, normalized_format)
    except AttributeError as e:
        raise ValueError(f"Unsupported RealSense color stream format '{normalized_format}'.") from e


def _decode_color_frame_to_rgb(color_frame: Any) -> NDArray[Any]:
    rs_module = _require_realsense()
    color_profile = color_frame.profile.as_video_stream_profile()
    width = int(color_profile.width())
    height = int(color_profile.height())
    frame_format = color_frame.profile.format()
    color_raw = np.asanyarray(color_frame.get_data())

    if frame_format == rs_module.format.rgb8:
        return np.ascontiguousarray(color_raw)

    if frame_format == rs_module.format.bgr8:
        return np.ascontiguousarray(cv2.cvtColor(color_raw, cv2.COLOR_BGR2RGB))

    if frame_format == rs_module.format.yuyv:
        color_raw = color_raw.view(np.uint8).reshape((height, width, 2))
        return np.ascontiguousarray(cv2.cvtColor(color_raw, cv2.COLOR_YUV2RGB_YUYV))

    raise RuntimeError(f"Unsupported RealSense color format: {frame_format}")


@dataclass(frozen=True)
class _StreamSpec:
    enabled: bool
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    format_name: str | None = None


@dataclass(frozen=True)
class _PipelineSpec:
    color: _StreamSpec
    depth: _StreamSpec

    def get(self, kind: StreamKind) -> _StreamSpec:
        return self.color if kind == "color" else self.depth

    @property
    def shared_fps(self) -> int | None:
        fps_values = {
            stream.fps for stream in (self.color, self.depth) if stream.enabled and stream.fps is not None
        }
        if len(fps_values) > 1:
            return None
        return next(iter(fps_values), None)

    def satisfies(self, required: "_PipelineSpec") -> bool:
        return _stream_satisfies(self.color, required.color) and _stream_satisfies(self.depth, required.depth)


@dataclass(frozen=True)
class _ClientRequest:
    kind: StreamKind
    fps: int | None
    width: int | None
    height: int | None
    color_stream_format: str | None = None
    warmup_s: float = 0.0


@dataclass(frozen=True)
class _FrameSnapshot:
    color: NDArray[Any] | None = None
    depth: NDArray[Any] | None = None


def _stream_satisfies(current: _StreamSpec, required: _StreamSpec) -> bool:
    if not required.enabled:
        return True
    if not current.enabled:
        return False
    if required.width is not None and current.width != required.width:
        return False
    if required.height is not None and current.height != required.height:
        return False
    if required.fps is not None and current.fps != required.fps:
        return False
    if required.format_name is not None and current.format_name != required.format_name:
        return False
    return True


class SharedRealSenseManager:
    _registry: dict[str, SharedRealSenseManager] = {}
    _registry_lock = Lock()

    def __init__(
        self,
        serial_number: str,
        model_traits: RealSenseModelTraits,
        device_name: str | None = None,
        usb_type_descriptor: str | None = None,
    ):
        self.serial_number = serial_number
        self.model_traits = model_traits
        self.device_name = device_name
        self.usb_type_descriptor = usb_type_descriptor

        self._lock = Lock()
        self._clients: dict[int, _ClientRequest] = {}

        self.pipeline: Any = None
        self.profile: Any = None
        self.current_spec: _PipelineSpec | None = None
        self.depth_scale = 0.001

        self.frame_lock = Lock()
        self.frame_condition = Condition(self.frame_lock)
        self.latest_snapshot: _FrameSnapshot | None = None
        self.frames_seq = 0
        self.last_frames_ts: float | None = None
        self._suppress_read_loop_warnings_until: float = 0.0

        self.thread: Thread | None = None
        self.stop_event = Event()

    @classmethod
    def get_or_create(
        cls,
        serial_number: str,
        model_traits: RealSenseModelTraits,
        device_name: str | None = None,
        usb_type_descriptor: str | None = None,
    ) -> SharedRealSenseManager:
        with cls._registry_lock:
            manager = cls._registry.get(serial_number)
            if manager is None:
                manager = cls(
                    serial_number=serial_number,
                    model_traits=model_traits,
                    device_name=device_name,
                    usb_type_descriptor=usb_type_descriptor,
                )
                cls._registry[serial_number] = manager
            else:
                if manager.model_traits.key != model_traits.key:
                    raise RuntimeError(
                        f"RealSense serial {serial_number} is already registered as "
                        f"{manager.model_traits.model_name}, but a {model_traits.model_name} camera was requested."
                    )
                manager.update_device_info(device_name=device_name, usb_type_descriptor=usb_type_descriptor)
            return manager

    @classmethod
    def drop_if_unused(cls, serial_number: str, manager: SharedRealSenseManager) -> None:
        with cls._registry_lock:
            if not manager.has_clients():
                current = cls._registry.get(serial_number)
                if current is manager:
                    cls._registry.pop(serial_number, None)

    def has_clients(self) -> bool:
        with self._lock:
            return bool(self._clients)

    def update_device_info(self, device_name: str | None, usb_type_descriptor: str | None) -> None:
        with self._lock:
            if device_name:
                self.device_name = device_name
            if usb_type_descriptor:
                self.usb_type_descriptor = usb_type_descriptor

    def connect_client(self, client_id: int, request: _ClientRequest) -> _PipelineSpec:
        with self._lock:
            if client_id in self._clients:
                if self.current_spec is None:
                    raise RuntimeError(f"Manager for {self.serial_number} lost its active pipeline state.")
                return self.current_spec

            desired_spec = self._build_desired_spec(extra_request=request)
            if self.current_spec is None or not self.current_spec.satisfies(desired_spec):
                self._restart_pipeline_locked(desired_spec)

            self._clients[client_id] = request
            self._extend_startup_warning_suppression(request.warmup_s)
            if self.current_spec is None:
                raise RuntimeError(f"Manager for {self.serial_number} failed to start a pipeline.")
            return self.current_spec

    def disconnect_client(self, client_id: int) -> None:
        should_drop = False
        with self._lock:
            self._clients.pop(client_id, None)
            if self._clients:
                return
            self._stop_pipeline_locked()
            should_drop = True

        if should_drop:
            self.drop_if_unused(self.serial_number, self)

    def get_latest_snapshot(self) -> _FrameSnapshot | None:
        with self.frame_lock:
            return self.latest_snapshot

    def get_latest_frames_and_seq(self) -> tuple[_FrameSnapshot | None, int]:
        with self.frame_lock:
            return self.latest_snapshot, self.frames_seq

    def wait_for_next_snapshot(self, last_seq: int, timeout_s: float) -> tuple[_FrameSnapshot | None, int]:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        with self.frame_condition:
            while True:
                if self.latest_snapshot is not None and self.frames_seq != last_seq:
                    return self.latest_snapshot, self.frames_seq

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, last_seq

                self.frame_condition.wait(timeout=remaining)

    def _extend_startup_warning_suppression(self, warmup_s: float) -> None:
        if warmup_s <= 0:
            return
        self._suppress_read_loop_warnings_until = max(
            self._suppress_read_loop_warnings_until,
            time.monotonic() + warmup_s + 0.5,
        )

    def get_output_stream_spec(self, kind: StreamKind) -> _StreamSpec:
        with self._lock:
            current_spec = self.current_spec

        if current_spec is None:
            raise RuntimeError(f"Manager for {self.serial_number} does not have an active pipeline.")

        return current_spec.get(kind)

    def get_status(self) -> dict[str, Any]:
        with self.frame_lock:
            last_frames_ts = self.last_frames_ts
            frames_seq = self.frames_seq

        with self._lock:
            current_spec = self.current_spec
            client_count = len(self._clients)
            device_name = self.device_name
            usb_type_descriptor = self.usb_type_descriptor

        return {
            "serial_number": self.serial_number,
            "device_name": device_name,
            "model": self.model_traits.model_name,
            "usb_type_descriptor": usb_type_descriptor,
            "has_pipeline": self.pipeline is not None,
            "clients": client_count,
            "frames_seq": frames_seq,
            "last_frames_age_s": None if last_frames_ts is None else (time.time() - last_frames_ts),
            "thread_alive": self.thread is not None and self.thread.is_alive(),
            "current_spec": current_spec,
        }

    def _build_desired_spec(self, extra_request: _ClientRequest | None = None) -> _PipelineSpec:
        requests = list(self._clients.values())
        if extra_request is not None:
            requests.append(extra_request)

        explicit_fps = {request.fps for request in requests if request.fps is not None}
        if len(explicit_fps) > 1:
            raise ValueError(
                f"RealSense {self.model_traits.model_name} cameras sharing serial {self.serial_number} "
                "must use the same fps. "
                f"Requested fps values: {sorted(explicit_fps)}"
            )

        sticky_fps = self.current_spec.shared_fps if self.current_spec is not None else None
        fps = next(iter(explicit_fps), sticky_fps)

        color_enabled = any(request.kind == "color" for request in requests)
        depth_enabled = any(request.kind == "depth" for request in requests)
        if self.current_spec is not None:
            color_enabled = color_enabled or self.current_spec.color.enabled
            depth_enabled = depth_enabled or self.current_spec.depth.enabled

        color_spec = self._build_stream_spec(kind="color", enabled=color_enabled, requests=requests, fps=fps)
        depth_spec = self._build_stream_spec(kind="depth", enabled=depth_enabled, requests=requests, fps=fps)
        return _PipelineSpec(color=color_spec, depth=depth_spec)

    def _build_stream_spec(
        self, kind: StreamKind, enabled: bool, requests: list[_ClientRequest], fps: int | None
    ) -> _StreamSpec:
        if not enabled:
            return _StreamSpec(enabled=False)

        current_stream = self.current_spec.get(kind) if self.current_spec is not None else _StreamSpec(False)

        widths = {request.width for request in requests if request.kind == kind and request.width is not None}
        heights = {request.height for request in requests if request.kind == kind and request.height is not None}
        if len(widths) > 1 or len(heights) > 1:
            raise ValueError(
                f"RealSense {self.model_traits.model_name} {kind} cameras sharing serial {self.serial_number} "
                "must use the same resolution."
            )

        width = next(iter(widths), current_stream.width if current_stream.enabled else None)
        height = next(iter(heights), current_stream.height if current_stream.enabled else None)
        format_name = None
        if kind == "color":
            formats = {
                request.color_stream_format
                for request in requests
                if request.kind == "color" and request.color_stream_format is not None
            }
            if len(formats) > 1:
                raise ValueError(
                    f"RealSense {self.model_traits.model_name} color cameras sharing serial {self.serial_number} "
                    "must use the same color stream format."
                )
            format_name = next(iter(formats), current_stream.format_name if current_stream.enabled else "rgb8")

        return _StreamSpec(enabled=True, width=width, height=height, fps=fps, format_name=format_name)

    def _restart_pipeline_locked(self, desired_spec: _PipelineSpec) -> None:
        self._stop_pipeline_locked()
        self._start_pipeline_locked(desired_spec)

    def _start_pipeline_locked(self, desired_spec: _PipelineSpec) -> None:
        rs_module = _require_realsense()
        pipeline = rs_module.pipeline()
        rs_config = self._create_rs_config(desired_spec)

        try:
            profile = pipeline.start(rs_config)
        except Exception as e:
            message = str(e)
            if "resource busy" in message.lower():
                raise RuntimeError(
                    f"{self.model_traits.model_name} is busy. Close other processes that may be using the camera "
                    "(for example another viewer, RealSense Viewer, ffmpeg, or a previous script)."
                ) from e
            raise ConnectionError(
                f"Failed to open RealSense {self.model_traits.model_name} camera {self.serial_number} "
                "with requested streams "
                f"{desired_spec}. The device or SDK rejected this profile. Run `lerobot-find-cameras realsense` "
                "to inspect the connected model, serial number, USB link, and default stream profile."
            ) from e

        actual_spec = self._extract_actual_spec(profile, desired_spec)

        self.pipeline = pipeline
        self.profile = profile
        self.current_spec = actual_spec
        self.depth_scale = self._extract_depth_scale(profile)

        with self.frame_condition:
            self.latest_snapshot = None
            self.frames_seq = 0
            self.last_frames_ts = None
            self.frame_condition.notify_all()

        usb_warning = _make_usb_warning(self.usb_type_descriptor)
        if usb_warning:
            logger.warning("RealSense %s (%s): %s", self.model_traits.model_name, self.serial_number, usb_warning)

        self.stop_event = Event()
        self.thread = Thread(
            target=self._read_loop,
            name=f"RealSense{self.model_traits.model_name}Manager[{self.serial_number}]_read_loop",
            daemon=True,
        )
        self.thread.start()

    def _create_rs_config(self, desired_spec: _PipelineSpec) -> Any:
        rs_module = _require_realsense()
        rs_config = rs_module.config()
        rs_config.enable_device(self.serial_number)
        self._enable_stream(rs_config, kind="depth", spec=desired_spec.depth)
        self._enable_stream(rs_config, kind="color", spec=desired_spec.color)
        return rs_config

    def _enable_stream(self, rs_config: Any, kind: StreamKind, spec: _StreamSpec) -> None:
        if not spec.enabled:
            return

        rs_module = _require_realsense()
        stream = rs_module.stream.color if kind == "color" else rs_module.stream.depth
        fmt = _get_rs_color_format(rs_module, spec.format_name) if kind == "color" else rs_module.format.z16
        fps = spec.fps if spec.fps is not None else 0

        if spec.width is None or spec.height is None:
            rs_config.enable_stream(stream, 0, 0, fmt, fps)
            return

        rs_config.enable_stream(stream, spec.width, spec.height, fmt, fps)

    def _extract_actual_spec(self, profile: Any, desired_spec: _PipelineSpec) -> _PipelineSpec:
        rs_module = _require_realsense()
        color_spec = _StreamSpec(enabled=False)
        depth_spec = _StreamSpec(enabled=False)

        if desired_spec.color.enabled:
            color_profile = profile.get_stream(rs_module.stream.color).as_video_stream_profile()
            color_spec = _StreamSpec(
                enabled=True,
                width=int(color_profile.width()),
                height=int(color_profile.height()),
                fps=int(color_profile.fps()),
                format_name=desired_spec.color.format_name,
            )

        if desired_spec.depth.enabled:
            depth_profile = profile.get_stream(rs_module.stream.depth).as_video_stream_profile()
            depth_spec = _StreamSpec(
                enabled=True,
                width=int(depth_profile.width()),
                height=int(depth_profile.height()),
                fps=int(depth_profile.fps()),
            )

        return _PipelineSpec(color=color_spec, depth=depth_spec)

    def _extract_depth_scale(self, profile: Any) -> float:
        try:
            device = profile.get_device()
            for sensor in device.query_sensors():
                if hasattr(sensor, "get_depth_scale"):
                    return float(sensor.get_depth_scale())
        except Exception:
            pass
        return 0.001

    def _read_loop(self) -> None:
        pipeline = self.pipeline
        stop_event = self.stop_event
        if pipeline is None or stop_event is None:
            return

        while not stop_event.is_set():
            try:
                ret, frames = pipeline.try_wait_for_frames(timeout_ms=100)
                if not ret or frames is None:
                    continue

                snapshot = self._build_frame_snapshot(frames)
                with self.frame_condition:
                    self.latest_snapshot = snapshot
                    self.frames_seq += 1
                    self.last_frames_ts = time.time()
                    self.frame_condition.notify_all()

            except Exception as e:
                if stop_event.is_set():
                    break
                error_text = str(e)
                if "frame_ref" in error_text:
                    logger.debug(
                        "Suppressing transient RealSense %s frame_ref exception for %s: %s",
                        self.model_traits.model_name,
                        self.serial_number,
                        e,
                    )
                    time.sleep(0.05)
                    continue
                startup_suppressed = (
                    time.monotonic() < self._suppress_read_loop_warnings_until or self.last_frames_ts == 0.0
                )
                if startup_suppressed:
                    logger.debug(
                        "Suppressing startup RealSense %s read loop exception for %s: %s",
                        self.model_traits.model_name,
                        self.serial_number,
                        e,
                    )
                else:
                    logger.warning(
                        "RealSense %s read loop exception for %s: %s",
                        self.model_traits.model_name,
                        self.serial_number,
                        e,
                    )
                time.sleep(0.05)

    def _build_frame_snapshot(self, frames: Any) -> _FrameSnapshot:
        color = None
        depth = None

        color_frame = frames.get_color_frame()
        if color_frame is not None:
            color = np.ascontiguousarray(_decode_color_frame_to_rgb(color_frame))

        depth_frame = frames.get_depth_frame()
        if depth_frame is not None:
            depth = np.ascontiguousarray(np.asanyarray(depth_frame.get_data()).copy())

        return _FrameSnapshot(color=color, depth=depth)

    def _stop_pipeline_locked(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception as e:
                logger.warning(
                    "Error stopping RealSense %s pipeline for %s: %s",
                    self.model_traits.model_name,
                    self.serial_number,
                    e,
                )

        self.thread = None
        self.stop_event = Event()
        self.pipeline = None
        self.profile = None
        self.current_spec = None
        self.depth_scale = 0.001

        with self.frame_condition:
            self.latest_snapshot = None
            self.frames_seq = 0
            self.last_frames_ts = None
            self.frame_condition.notify_all()

class SharedRealSenseBaseCamera(Camera):
    KIND: StreamKind
    MODEL_TRAITS: RealSenseModelTraits

    def __init__(
        self,
        config: (
            RealSenseD405ColorCameraConfig
            | RealSenseD405DepthCameraConfig
            | RealSenseD435iColorCameraConfig
            | RealSenseD435iDepthCameraConfig
        ),
    ):
        super().__init__(config)
        self.config = config
        self.serial_number, self.device_name = _resolve_serial_number(
            config.serial_number_or_name, supported_traits=(self.MODEL_TRAITS,)
        )
        self.manager: SharedRealSenseManager | None = None
        self.connected = False

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.new_frame_event = Event()
        self._last_frame_ts: float | None = None
        self._last_pipeline_seq = -1

        self.rotation = get_cv2_rotation(config.rotation)
        self.capture_width = self.width
        self.capture_height = self.height

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"

    @property
    def is_connected(self) -> bool:
        return self.connected

    @classmethod
    def find_cameras(cls) -> list[dict[str, Any]]:
        return [
            {
                "name": device["name"],
                "model": device["model"],
                "type": "RealSense",
                "id": device["serial_number"],
                "serial_number": device["serial_number"],
                "firmware_version": device["firmware_version"],
                "usb_type_descriptor": device["usb_type_descriptor"],
                "usb_warning": device["usb_warning"],
                "product_line": device["product_line"],
                "product_id": device["product_id"],
                "default_stream_profile": device["default_stream_profile"],
            }
            for device in _query_realsense_devices()
            if _matches_supported_model(device, (cls.MODEL_TRAITS,))
        ]

    def connect(self, warmup: bool = True) -> None:
        if self.connected:
            return

        device_info = _find_realsense_device_info(self.serial_number, supported_traits=(self.MODEL_TRAITS,))
        self.device_name = device_info["name"]
        self.manager = SharedRealSenseManager.get_or_create(
            self.serial_number,
            model_traits=self.MODEL_TRAITS,
            device_name=self.device_name,
            usb_type_descriptor=device_info["usb_type_descriptor"],
        )

        try:
            spec = self.manager.connect_client(client_id=id(self), request=self._build_client_request())
            self._apply_stream_settings(spec.get(self.KIND))
        except Exception:
            if self.manager is not None:
                SharedRealSenseManager.drop_if_unused(self.serial_number, self.manager)
            self.manager = None
            raise

        self.connected = True

        if warmup:
            warmup_s = float(getattr(self.config, "warmup_s", 0) or 0)
            deadline = time.time() + warmup_s
            got_frame = False
            last_error: Exception | None = None
            while time.time() < deadline:
                try:
                    frame = self.read()
                    if frame is not None:
                        got_frame = True
                        break
                except Exception as e:
                    last_error = e
                time.sleep(0.05)

            if not got_frame and warmup_s > 0:
                startup_suppressed = self.manager is not None and (
                    time.monotonic() < self.manager._suppress_read_loop_warnings_until
                    or self.manager.last_frames_ts == 0.0
                )
                if startup_suppressed and last_error is not None:
                    logger.debug("Warmup timed out for %s. Last error: %s", self, last_error)
                elif startup_suppressed:
                    logger.debug("Warmup timed out for %s.", self)
                elif last_error is not None:
                    logger.warning("Warmup timed out for %s. Last error: %s", self, last_error)
                else:
                    logger.warning("Warmup timed out for %s.", self)

    def _build_client_request(self) -> _ClientRequest:
        color_stream_format = getattr(self.config, "color_stream_format", None) if self.KIND == "color" else None
        return _ClientRequest(
            kind=self.KIND,
            fps=self.config.fps,
            width=self.config.width,
            height=self.config.height,
            color_stream_format=color_stream_format,
            warmup_s=float(getattr(self.config, "warmup_s", 0) or 0),
        )

    def _apply_stream_settings(self, stream_spec: _StreamSpec) -> None:
        if not stream_spec.enabled or stream_spec.width is None or stream_spec.height is None:
            raise RuntimeError(f"{self} failed to resolve active stream settings for {self.KIND}.")

        self.fps = stream_spec.fps
        self.capture_width = stream_spec.width
        self.capture_height = stream_spec.height

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.width = stream_spec.height
            self.height = stream_spec.width
        else:
            self.width = stream_spec.width
            self.height = stream_spec.height

    def _sync_stream_settings(self) -> None:
        if not self.connected or self.manager is None:
            return

        stream_spec = self.manager.get_output_stream_spec(self.KIND)
        if (
            stream_spec.enabled
            and (
                stream_spec.width != self.capture_width
                or stream_spec.height != self.capture_height
                or stream_spec.fps != self.fps
            )
        ):
            self._apply_stream_settings(stream_spec)

    def _read_from_snapshot(self, snapshot: _FrameSnapshot, color_mode: ColorMode | None = None) -> NDArray[Any] | None:
        raise NotImplementedError

    def _process_snapshot(self, snapshot: _FrameSnapshot) -> NDArray[Any] | None:
        frame = self._read_from_snapshot(snapshot)
        if frame is None:
            return None

        with self.frame_lock:
            self.latest_frame = frame
        self._last_frame_ts = time.time()
        return frame

    def _read_loop(self) -> None:
        if self.manager is None:
            raise RuntimeError(f"{self}: manager must be initialized before starting the read loop.")

        while not (self.stop_event and self.stop_event.is_set()):
            try:
                snapshot, seq = self.manager.wait_for_next_snapshot(self._last_pipeline_seq, timeout_s=0.5)
                if snapshot is None or seq == self._last_pipeline_seq:
                    continue

                self._last_pipeline_seq = seq
                frame = self._process_snapshot(snapshot)
                if frame is None:
                    continue

                self.new_frame_event.set()
            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning("Error reading frame in background thread for %s: %s", self, e)
                time.sleep(0.05)

    def _start_read_thread(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self.__class__.__name__}_read_loop", daemon=True)
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 5000) -> NDArray[Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.manager is None:
            raise RuntimeError(f"{self} does not have an active RealSense manager.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            manager_status = self.manager.get_status()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self}. "
                f"Manager status: {manager_status}"
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Event set but no frame available for {self}.")

        return frame

    def disconnect(self) -> None:
        if not self.connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} already disconnected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self.manager is not None:
            self.manager.disconnect_client(id(self))

        self.connected = False
        self.latest_frame = None
        self.new_frame_event.clear()
        self._last_pipeline_seq = -1
        self.manager = None

    def _require_snapshot(self) -> _FrameSnapshot | None:
        if not self.connected or self.manager is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._sync_stream_settings()
        return self.manager.get_latest_snapshot()


class SharedRealSenseColorCamera(SharedRealSenseBaseCamera):
    KIND = "color"

    def __init__(self, config: RealSenseD405ColorCameraConfig | RealSenseD435iColorCameraConfig):
        super().__init__(config)
        self.color_mode = config.color_mode

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any] | None:
        snapshot = self._require_snapshot()
        if snapshot is None:
            return None
        return self._read_from_snapshot(snapshot, color_mode=color_mode)

    def _read_from_snapshot(
        self, snapshot: _FrameSnapshot, color_mode: ColorMode | None = None
    ) -> NDArray[Any] | None:
        self._sync_stream_settings()
        if snapshot.color is None:
            return None
        return self._postprocess_color_image(snapshot.color, color_mode=color_mode)

    def _postprocess_color_image(
        self, image_rgb: NDArray[Any], color_mode: ColorMode | None = None
    ) -> NDArray[Any]:
        if color_mode and color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid requested color mode '{color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise RuntimeError(f"{self} frame is expected to be HxWx3, but got shape={image_rgb.shape}.")

        height, width, _ = image_rgb.shape
        if height != self.capture_height or width != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={width} or height={height} do not match configured width="
                f"{self.capture_width} or height={self.capture_height}."
            )

        requested_color_mode = color_mode or self.color_mode
        processed_image = image_rgb
        if requested_color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image


class SharedRealSenseDepthCamera(SharedRealSenseBaseCamera):
    KIND = "depth"

    def __init__(self, config: RealSenseD405DepthCameraConfig | RealSenseD435iDepthCameraConfig):
        super().__init__(config)
        self.max_depth_m = (
            float(config.max_depth_m)
            if self.MODEL_TRAITS.supports_max_depth_m and hasattr(config, "max_depth_m")
            else None
        )
        self.depth_alpha = float(getattr(config, "depth_alpha", 0.03))

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any] | None:
        del color_mode
        snapshot = self._require_snapshot()
        if snapshot is None:
            return None
        return self._read_from_snapshot(snapshot)

    def _read_from_snapshot(self, snapshot: _FrameSnapshot, color_mode: ColorMode | None = None) -> NDArray[Any] | None:
        del color_mode
        self._sync_stream_settings()
        if snapshot.depth is None:
            return None
        return self._postprocess_depth_image(snapshot.depth)

    def _postprocess_depth_image(self, depth_map: NDArray[Any]) -> NDArray[Any]:
        if depth_map.ndim != 2:
            raise RuntimeError(f"{self} depth frame is expected to be HxW, but got shape={depth_map.shape}.")

        height, width = depth_map.shape
        if height != self.capture_height or width != self.capture_width:
            raise RuntimeError(
                f"{self} depth frame width={width} or height={height} do not match configured width="
                f"{self.capture_width} or height={self.capture_height}."
            )

        depth_u16 = depth_map.astype(np.uint16, copy=False)
        valid_mask = depth_u16 > 0
        if self.max_depth_m is not None:
            depth_scale = self.manager.depth_scale if self.manager is not None else 0.001
            max_depth_raw = int(self.max_depth_m / depth_scale)
            valid_mask &= depth_u16 <= max_depth_raw

        depth_8u = cv2.convertScaleAbs(depth_u16, alpha=self.depth_alpha)
        depth_rgb = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
        depth_rgb[~valid_mask] = 0

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            depth_rgb = cv2.rotate(depth_rgb, self.rotation)

        return depth_rgb


class RealSenseD405ColorCamera(SharedRealSenseColorCamera):
    MODEL_TRAITS = D405_MODEL_TRAITS


class RealSenseD405DepthCamera(SharedRealSenseDepthCamera):
    MODEL_TRAITS = D405_MODEL_TRAITS


class RealSenseD435iColorCamera(SharedRealSenseColorCamera):
    MODEL_TRAITS = D435I_MODEL_TRAITS


class RealSenseD435iDepthCamera(SharedRealSenseDepthCamera):
    MODEL_TRAITS = D435I_MODEL_TRAITS


# Compatibility aliases for local tools that still import the old manager names.
RealSenseD405Manager = SharedRealSenseManager
RealSenseD435iManager = SharedRealSenseManager
