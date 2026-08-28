from __future__ import annotations

"""
Orbbec 相机接入层。

这个文件提供 Orbbec 的 split camera 接入层：

- `OrbbecPipeline` 负责唯一的底层 SDK pipeline 和唯一的采集线程
- `OrbbecColorCamera` / `OrbbecDepthCamera` 是暴露给上层框架的逻辑相机
- 设计目标和当前的 RealSense split 逻辑保持一致：
  - 底层硬件连接只有一份
  - 上层 color/depth 是两个逻辑实例
  - `find-cameras` / `teleoperate` / `record` 都直接使用 split camera
"""

import logging
import time
from threading import Condition, Event, Lock, Thread
from typing import Any, Optional

import cv2
import numpy as np
import pyorbbecsdk as ob

from lerobot.utils.errors import DeviceNotConnectedError
from ..camera import Camera
from ..configs import CameraConfig, ColorMode
from ..utils import get_cv2_rotation
from .configuration_orbbec import (
    OrbbecColorCameraConfig,
    OrbbecDepthCameraConfig,
)

logger = logging.getLogger(__name__)


# -------------------------
# 设备信息查询辅助函数
# -------------------------

def _get_orbbec_info_value(device_info: Any, method_name: str) -> Any | None:
    """安全调用 device_info 的 getter。

    Orbbec SDK 在不同版本/不同设备上暴露的方法并不完全一致，这里统一做
    `getattr + try/except`，避免枚举设备信息时因为单个字段失败而中断整次发现流程。
    """
    getter = getattr(device_info, method_name, None)
    if getter is None:
        return None

    try:
        return getter()
    except Exception:
        return None


def _get_orbbec_list_value(device_list: Any, method_name: str, index: int) -> Any | None:
    """安全调用 device_list 上按索引读取元数据的方法。"""
    getter = getattr(device_list, method_name, None)
    if getter is None:
        return None

    try:
        return getter(index)
    except Exception:
        return None


def _get_orbbec_default_stream_profiles(device: Any) -> dict[str, dict[str, Any]]:
    """提取设备各传感器的默认 stream profile，用于 `find-cameras` 展示。"""
    camera_info: dict[str, dict[str, Any]] = {}
    first_stream_profile: dict[str, Any] | None = None

    try:
        sensor_list = device.get_sensor_list()
    except Exception as e:
        logger.debug("Failed to query Orbbec sensor list: %s", e)
        return camera_info

    for index in range(sensor_list.get_count()):
        try:
            sensor = sensor_list.get_sensor_by_index(index)
            sensor_type = getattr(sensor.get_type(), "name", str(sensor.get_type()))
            profiles = sensor.get_stream_profile_list()
        except Exception as e:
            logger.debug("Failed to query Orbbec stream profiles: %s", e)
            continue

        try:
            profile = profiles.get_default_video_stream_profile()
            vprofile = profile.as_video_stream_profile()
            stream_name = sensor_type.removesuffix("_SENSOR").replace("_", " ").title()
            stream_info = {
                "stream_type": stream_name,
                "format": vprofile.get_format().name,
                "width": vprofile.get_width(),
                "height": vprofile.get_height(),
                "fps": vprofile.get_fps(),
            }

            if first_stream_profile is None:
                first_stream_profile = stream_info

            stream_key = stream_name.lower().replace(" ", "_")
            camera_info[f"default_{stream_key}_stream_profile"] = stream_info
        except Exception as e:
            logger.debug("Failed to inspect Orbbec default stream profile: %s", e)

    if "default_color_stream_profile" in camera_info:
        camera_info["default_stream_profile"] = camera_info["default_color_stream_profile"]
    elif "default_depth_stream_profile" in camera_info:
        camera_info["default_stream_profile"] = camera_info["default_depth_stream_profile"]
    elif first_stream_profile is not None:
        camera_info["default_stream_profile"] = first_stream_profile

    return camera_info


def _describe_orbbec_device(device: Any) -> dict[str, Any]:
    """从真实 device 对象读取较完整的设备描述信息。"""
    try:
        device_info = device.get_device_info()
    except Exception as e:
        logger.debug("Failed to read Orbbec device info: %s", e)
        device_info = None

    name = _get_orbbec_info_value(device_info, "get_name") if device_info is not None else None
    serial_number = (
        _get_orbbec_info_value(device_info, "get_serial_number") if device_info is not None else None
    )
    uid = _get_orbbec_info_value(device_info, "get_uid") if device_info is not None else None

    camera_info: dict[str, Any] = {
        "name": name or "Orbbec Camera",
        "type": "Orbbec",
        "id": serial_number or uid or "unknown",
    }

    metadata_fields = {
        "serial_number": "get_serial_number",
        "firmware_version": "get_firmware_version",
        "connection_type": "get_connection_type",
        "hardware_version": "get_hardware_version",
        "uid": "get_uid",
        "pid": "get_pid",
        "vid": "get_vid",
    }
    for output_key, method_name in metadata_fields.items():
        if device_info is None:
            continue
        value = _get_orbbec_info_value(device_info, method_name)
        if value not in (None, ""):
            camera_info[output_key] = value

    camera_info.update(_get_orbbec_default_stream_profiles(device))
    return camera_info


def _describe_orbbec_device_from_list(device_list: Any, index: int) -> dict[str, Any]:
    """从 device_list 按索引读取兜底元数据。

    这个路径拿到的信息比真实 `device` 对象少，但在某些 SDK/权限异常下依然能工作，
    因此这里作为设备发现的 fallback。
    """
    serial_number = _get_orbbec_list_value(device_list, "get_device_serial_number_by_index", index)
    uid = _get_orbbec_list_value(device_list, "get_device_uid_by_index", index)

    camera_info: dict[str, Any] = {
        "name": _get_orbbec_list_value(device_list, "get_device_name_by_index", index) or "Orbbec Camera",
        "type": "Orbbec",
        "id": serial_number or uid or str(index),
    }

    metadata_fields = {
        "serial_number": "get_device_serial_number_by_index",
        "connection_type": "get_device_connection_type_by_index",
        "uid": "get_device_uid_by_index",
        "pid": "get_device_pid_by_index",
        "vid": "get_device_vid_by_index",
    }
    for output_key, method_name in metadata_fields.items():
        value = _get_orbbec_list_value(device_list, method_name, index)
        if value not in (None, ""):
            camera_info[output_key] = value

    return camera_info


def _query_orbbec_cameras() -> list[dict[str, Any]]:
    """枚举当前机器上的 Orbbec 设备并尽量补齐可展示的信息。"""
    found_cameras_info: list[dict[str, Any]] = []
    context = ob.Context()
    device_list = context.query_devices()

    for index in range(device_list.get_count()):
        fallback_info = _describe_orbbec_device_from_list(device_list, index)
        try:
            device = device_list.get_device_by_index(index)
            camera_info = _describe_orbbec_device(device)
        except Exception as e:
            logger.debug("Falling back to Orbbec device-list metadata for index %s: %s", index, e)
            camera_info = {}

        merged_camera_info = dict(fallback_info)
        merged_camera_info.update(camera_info)
        found_cameras_info.append(merged_camera_info)

    return found_cameras_info


def find_orbbec_cameras() -> list[dict[str, Any]]:
    return _query_orbbec_cameras()


def _find_orbbec_device_info(serial_number_or_name: str) -> dict[str, Any]:
    devices = _query_orbbec_cameras()
    matches = [device for device in devices if device.get("serial_number") == serial_number_or_name]
    if not matches:
        matches = [device for device in devices if device.get("name") == serial_number_or_name]

    if not matches:
        available = [f'{device.get("name", "Orbbec")} ({device.get("serial_number", device.get("id"))})' for device in devices]
        raise ValueError(
            f"No Orbbec camera found for '{serial_number_or_name}'. Available devices: {available}"
        )

    if len(matches) > 1:
        serials = [str(device.get("serial_number", device.get("id"))) for device in matches]
        raise ValueError(
            f"Multiple Orbbec cameras found with name '{serial_number_or_name}'. "
            f"Please use the serial number instead. Found serials: {serials}"
        )

    return matches[0]


def _resolve_orbbec_serial_number(serial_number_or_name: str) -> tuple[str, str | None]:
    device_info = _find_orbbec_device_info(serial_number_or_name)
    serial_number = str(device_info.get("serial_number") or device_info.get("id") or serial_number_or_name)
    return serial_number, device_info.get("name")


def _find_orbbec_device_by_serial(serial_number: str) -> Any:
    context = ob.Context()
    device_list = context.query_devices()

    getter = getattr(device_list, "get_device_by_serial_number", None)
    if getter is not None:
        try:
            return getter(serial_number)
        except Exception:
            pass

    for index in range(device_list.get_count()):
        device = device_list.get_device_by_index(index)
        try:
            device_info = device.get_device_info()
        except Exception:
            device_info = None
        current_serial = _get_orbbec_info_value(device_info, "get_serial_number")
        if str(current_serial) == serial_number:
            return device

    raise ValueError(f"Failed to locate Orbbec device with serial {serial_number}.")


class OrbbecPipeline:
    """
    全局唯一 Orbbec pipeline 管理者（Singleton）。

    职责（清晰分界）：
      - 聚合多个 Camera 的能力声明（color / depth）
      - 协商 pipeline 级参数（fps / width / height）的一致性
      - 负责启动 Orbbec SDK 的 pipeline（.start）并运行一个 *唯一的* 后台采集线程，
        该线程是唯一调用 wait_for_frames() 的位置。
      - 提供线程安全的 get_latest_frames() 供 Camera 视图读取（非阻塞）。

    设计原则：
      - 参数协商与启动为原子操作（由类级锁保护）
      - 只允许一个 producer（pipeline._read_loop）写 latest_frames
      - Camera 只读 latest_frames（通过 frame_lock）
    """

    _registry: dict[str, "OrbbecPipeline"] = {}
    _lock = Lock()  # 用于 registry 与参数协商保护

    def __init__(self, serial_number: str, device_name: str | None = None):
        self.serial_number = serial_number
        self.device_name = device_name
        # Orbbec SDK 对象（延后创建）
        self.pipeline = None
        self.config = None
        self.device = None
        self.started = False
        self._started_enable_color = False
        self._started_enable_depth = False

        # 能力声明（外部 register_camera 设置）
        self.enable_color = False
        self.enable_depth = False

        # 协商后确定的 pipeline 参数
        # 约束：同一个 pipeline 下，fps 必须一致；但 color/depth 分辨率允许不同
        self.fps: Optional[int] = None
        self.color_width: Optional[int] = None
        self.color_height: Optional[int] = None
        self.depth_width: Optional[int] = None
        self.depth_height: Optional[int] = None

        # 帧缓存（保护 latest_frames 的锁）
        self.frame_lock = Lock()
        self.frame_condition = Condition(self.frame_lock)
        self.latest_frames = None  # SDK frames object (完整的一帧组)
        self.frames_seq: int = 0
        self.last_frames_ts: float | None = None
        self.consecutive_wait_failures: int = 0
        self.total_wait_failures: int = 0
        self.total_wait_exceptions: int = 0
        self.last_wait_exception: str | None = None
        self._last_status_log_ts: float = 0.0

        # 采集线程控制
        self.thread: Optional[Thread] = None
        self.stop_event: Event = Event()
        self._clients: dict[int, str] = {}

    @classmethod
    def get_or_create(cls, serial_number: str, device_name: str | None = None) -> "OrbbecPipeline":
        with cls._lock:
            manager = cls._registry.get(serial_number)
            if manager is None:
                manager = cls(serial_number=serial_number, device_name=device_name)
                cls._registry[serial_number] = manager
            elif device_name:
                manager.device_name = device_name
            return manager

    @classmethod
    def drop_if_unused(cls, serial_number: str, manager: "OrbbecPipeline") -> None:
        with cls._lock:
            # 这里已经持有与 client 注册/注销同一把锁，不能再调 has_clients()，
            # 否则会对同一把非可重入锁再次加锁，导致 disconnect 路径卡死。
            if manager._clients:
                return
            current = cls._registry.get(serial_number)
            if current is manager:
                cls._registry.pop(serial_number, None)

    def has_clients(self) -> bool:
        with self._lock:
            return bool(self._clients)

    def _register_camera_locked(self, cfg: "CameraConfig", kind: str):
        """
        注册一个 Camera 的需求到 pipeline（线程安全）。

        - cfg: Camera 的配置对象（包含 fps/width/height/warmup_s 等）
        - kind: "color" 或 "depth"
        """
        # 能力声明
        if kind == "color":
            self.enable_color = True
        elif kind == "depth":
            self.enable_depth = True
        else:
            raise ValueError(f"Unknown camera kind: {kind}")

        # 协商 fps/width/height：第一次写入后固定，任何不一致都会抛错
        # 协商 fps：第一次写入后固定，不一致则抛错（同一 pipeline 下必须一致）
        v_fps = getattr(cfg, "fps", None)
        if v_fps is not None:
            if self.fps is None:
                self.fps = v_fps
            elif self.fps != v_fps:
                raise ValueError(
                    f"Pipeline parameter conflict on `fps`: "
                    f"{self.fps} (existing) vs {v_fps} (from {kind})"
                )

        # 分辨率允许 color/depth 不同，但同一种流（color 或 depth）内部必须一致
        v_w = getattr(cfg, "width", None)
        v_h = getattr(cfg, "height", None)
        if kind == "color":
            if v_w is not None:
                if self.color_width is None:
                    self.color_width = v_w
                elif self.color_width != v_w:
                    raise ValueError(
                        f"Pipeline parameter conflict on `color_width`: "
                        f"{self.color_width} (existing) vs {v_w} (from {kind})"
                    )
            if v_h is not None:
                if self.color_height is None:
                    self.color_height = v_h
                elif self.color_height != v_h:
                    raise ValueError(
                        f"Pipeline parameter conflict on `color_height`: "
                        f"{self.color_height} (existing) vs {v_h} (from {kind})"
                    )
        elif kind == "depth":
            if v_w is not None:
                if self.depth_width is None:
                    self.depth_width = v_w
                elif self.depth_width != v_w:
                    raise ValueError(
                        f"Pipeline parameter conflict on `depth_width`: "
                        f"{self.depth_width} (existing) vs {v_w} (from {kind})"
                    )
            if v_h is not None:
                if self.depth_height is None:
                    self.depth_height = v_h
                elif self.depth_height != v_h:
                    raise ValueError(
                        f"Pipeline parameter conflict on `depth_height`: "
                        f"{self.depth_height} (existing) vs {v_h} (from {kind})"
                    )

    def connect_client(self, client_id: int, cfg: "CameraConfig", kind: str) -> None:
        with self._lock:
            if client_id in self._clients:
                return

            self._register_camera_locked(cfg, kind)
            self._clients[client_id] = kind
            self._start_locked()

    def disconnect_client(self, client_id: int) -> None:
        with self._lock:
            self._clients.pop(client_id, None)
            if self._clients:
                return
            self._stop_locked(stop_pipeline=True, reset_config=True)

    def start(self):
        with self._lock:
            self._start_locked()

    def _start_locked(self):
        """
        启动 Orbbec pipeline 以及后台采集线程（只允许一次，线程安全）。

        重要校验：
          - 如果启用了 color 或 depth 且 width/fps/height 为 None -> 报错
            （避免 SDK 在缺少必需参数时崩溃或使用未定义行为）
        """
        # 如果 pipeline 已启动，但后续又注册了新的 stream 需求（例如先 connect color 再 connect depth），
        # 则必须重启 pipeline 才能让新 stream 生效。否则对应 read() 会一直返回 None，async_read 会持续超时。
        if self.started:
            if (
                self._started_enable_color == self.enable_color
                and self._started_enable_depth == self.enable_depth
            ):
                return
            logger.info(
                "Restarting Orbbec pipeline to apply stream requirements: "
                f"color {self._started_enable_color}->{self.enable_color}, "
                f"depth {self._started_enable_depth}->{self.enable_depth}"
            )
            self._restart_locked()

        # 在启用任一路流的情况下，fps 必须指定；分辨率需针对开启的流分别指定
        if (self.enable_color or self.enable_depth) and self.fps is None:
            raise RuntimeError("Pipeline start requires `fps` to be set by camera configs.")
        if self.enable_color and (self.color_width is None or self.color_height is None):
            raise RuntimeError(
                "Pipeline start requires `width`/`height` for color stream to be set by camera configs."
            )
        if self.enable_depth and (self.depth_width is None or self.depth_height is None):
            raise RuntimeError(
                "Pipeline start requires `width`/`height` for depth stream to be set by camera configs."
            )

        logger.info(
            "Starting Orbbec pipeline with "
            f"enable_color={self.enable_color}, enable_depth={self.enable_depth}, "
            f"color={self.color_width}x{self.color_height}, "
            f"depth={self.depth_width}x{self.depth_height}, fps={self.fps}"
        )

        # 创建 SDK 对象并启用对应流
        self.device = _find_orbbec_device_by_serial(self.serial_number)
        self.pipeline = ob.Pipeline(self.device)
        self.config = ob.Config()

        if self.enable_depth:
            profiles = self.pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
            depth_profile = profiles.get_video_stream_profile(
                self.depth_width, self.depth_height, ob.OBFormat.Y16, self.fps
            )
            logger.info(
                "Enabled DEPTH stream "
                f"{self.depth_width}x{self.depth_height}@{self.fps} format=Y16"
            )
            self.config.enable_stream(depth_profile)

        if self.enable_color:
            profiles = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            color_profile = profiles.get_video_stream_profile(
                self.color_width, self.color_height, ob.OBFormat.RGB, self.fps
            )
            logger.info(
                "Enabled COLOR stream "
                f"{self.color_width}x{self.color_height}@{self.fps} format=RGB"
            )
            self.config.enable_stream(color_profile)

        # 启动 SDK pipeline（阻塞/异常可能来自 SDK，自行处理）
        self.pipeline.start(self.config)
        self.started = True
        self._started_enable_color = self.enable_color
        self._started_enable_depth = self.enable_depth
        self.frames_seq = 0
        self.last_frames_ts = None
        self.consecutive_wait_failures = 0
        self.total_wait_failures = 0
        self.total_wait_exceptions = 0
        self.last_wait_exception = None

        # 启动唯一的后台采集线程（daemon）
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name="OrbbecPipeline_read_loop", daemon=True)
        self.thread.start()

    def _restart_locked(self) -> None:
        """
        需在持有 self._lock 的情况下调用。
        停止现有采集线程与 SDK pipeline，然后清空状态，以便后续重新 start()。
        """
        self.stop_event.set()

        # 先停 SDK pipeline，尽量把阻塞中的 wait_for_frames() 立刻唤醒，避免 join 卡在 SDK 内部。
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception as e:
                logger.warning(f"Error stopping pipeline during restart: {e}")

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("Orbbec read loop thread did not stop cleanly during restart.")

        self.thread = None
        self.stop_event = Event()

        self.pipeline = None
        self.config = None
        self.started = False
        self._started_enable_color = False
        self._started_enable_depth = False

        with self.frame_lock:
            self.latest_frames = None
            self.frames_seq = 0
            self.last_frames_ts = None

        self.consecutive_wait_failures = 0
        self.total_wait_failures = 0
        self.total_wait_exceptions = 0
        self.last_wait_exception = None

    def _read_loop(self):
        """
        唯一允许调用 wait_for_frames() 的地方（producer）。
        将 SDK frames 原子地写入 latest_frames（通过 frame_lock 保护）。
        保证了 color/depth 来自同一 frames 对象。
        """
        while not self.stop_event.is_set():
            try:
                # wait_for_frames 超时不宜过长，否则 stop/restart 不灵敏（这里设为 1000ms）
                frames = self.pipeline.wait_for_frames(1000)
                if frames is None:
                    self.consecutive_wait_failures += 1
                    self.total_wait_failures += 1
                    now = time.time()
                    if now - self._last_status_log_ts > 2.0:
                        self._last_status_log_ts = now
                        logger.warning(
                            "Orbbec pipeline wait_for_frames() returned None "
                            f"(consecutive={self.consecutive_wait_failures}, total={self.total_wait_failures})."
                        )
                    continue

                self.consecutive_wait_failures = 0
                now = time.time()
                with self.frame_lock:
                    # 整体替换 latest_frames，读者通过 frame_lock 读取
                    self.latest_frames = frames
                    self.frames_seq += 1
                    self.last_frames_ts = now
                    self.frame_condition.notify_all()
            except Exception as e:
                # 记录并继续。不要把线程抛出，否则会终止采集。
                self.total_wait_exceptions += 1
                self.last_wait_exception = repr(e)
                now = time.time()
                if now - self._last_status_log_ts > 2.0:
                    self._last_status_log_ts = now
                    logger.warning(
                        "Pipeline read loop exception "
                        f"(exceptions={self.total_wait_exceptions}, consecutive_none={self.consecutive_wait_failures}): {e}"
                    )
                continue

    def get_latest_frames(self):
        """
        提供给 Camera 的只读接口（线程安全）。
        返回 SDK frames 对象（可能为 None）。
        """
        with self.frame_lock:
            return self.latest_frames

    def get_latest_frames_and_seq(self):
        """
        返回 (latest_frames, frames_seq) 的一致快照（线程安全）。
        用于 Camera 侧避免对同一帧组重复做后处理，从而避免 CPU 空转导致帧率下降。
        """
        with self.frame_lock:
            return self.latest_frames, self.frames_seq

    def wait_for_next_frames(self, last_seq: int, timeout_s: float) -> tuple[Any, int]:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        with self.frame_condition:
            while True:
                if self.latest_frames is not None and self.frames_seq != last_seq:
                    return self.latest_frames, self.frames_seq

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, last_seq

                self.frame_condition.wait(timeout=remaining)

    def get_status(self) -> dict[str, Any]:
        with self.frame_lock:
            last_ts = self.last_frames_ts
            seq = self.frames_seq
        age_s = None if last_ts is None else (time.time() - last_ts)
        return {
            "serial_number": self.serial_number,
            "device_name": self.device_name,
            "started": self.started,
            "enable_color": self.enable_color,
            "enable_depth": self.enable_depth,
            "fps": self.fps,
            "color_width": self.color_width,
            "color_height": self.color_height,
            "depth_width": self.depth_width,
            "depth_height": self.depth_height,
            "frames_seq": seq,
            "last_frames_age_s": age_s,
            "consecutive_wait_failures": self.consecutive_wait_failures,
            "total_wait_failures": self.total_wait_failures,
            "total_wait_exceptions": self.total_wait_exceptions,
            "last_wait_exception": self.last_wait_exception,
            "thread_alive": (self.thread is not None and self.thread.is_alive()),
        }

    def stop(self, stop_pipeline: bool = True, reset_config: bool = False):
        with self._lock:
            self._stop_locked(stop_pipeline=stop_pipeline, reset_config=reset_config)

    def _stop_locked(self, stop_pipeline: bool = True, reset_config: bool = False):
        """
        停止采集线程并（可选）停止底层 SDK pipeline。
        """
        self.stop_event.set()

        # 和 restart 一样，优先 stop pipeline 来唤醒 SDK 阻塞读取，降低退出阶段卡死概率。
        if stop_pipeline and self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                logger.warning("Error stopping pipeline.")

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("Orbbec read loop thread did not stop cleanly.")

        self.thread = None
        self.stop_event = Event()
        self.started = False
        self._started_enable_color = False
        self._started_enable_depth = False
        with self.frame_lock:
            self.latest_frames = None
            self.frames_seq = 0
            self.last_frames_ts = None
            self.frame_condition.notify_all()

        if stop_pipeline and self.pipeline is not None:
            self.pipeline = None
            self.config = None
            self.device = None

        if reset_config:
            self.enable_color = False
            self.enable_depth = False
            self.fps = None
            self.color_width = None
            self.color_height = None
            self.depth_width = None
            self.depth_height = None


# -------------------------
# Camera 层（Base + Color + Depth）
# -------------------------

class OrbbecBaseCamera(Camera):
    """
    Orbbec 逻辑相机基类。

    职责：
      - 将自身 config 注册到 OrbbecPipeline
      - 管理视图级的后台线程（_read_loop / async_read）
      - 子类只需实现 read()（从 pipeline.get_latest_frames() 里提取视图并后处理）
    """

    KIND = None  # 子类必须覆盖为 "color" 或 "depth"

    def __init__(self, config: "CameraConfig"):
        super().__init__(config)
        self.config = config
        self.serial_number, self.device_name = _resolve_orbbec_serial_number(config.serial_number_or_name)
        self.pipeline = OrbbecPipeline.get_or_create(self.serial_number, self.device_name)
        self.connected = False

        self._last_frame_ts: float | None = None
        self._last_pipeline_seq: int = -1

        # 方便的常用值
        self.rotation = get_cv2_rotation(config.rotation)

    @property
    def is_connected(self) -> bool:
        return self.connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return find_orbbec_cameras()

    def connect(self, warmup: bool = True):
        """
        注册需求并启动 pipeline（pipeline.start 由 pipeline 管理保证只会执行一次）。
        这里也做 warmup：等待当前逻辑相机能读到第一帧（直到 warmup 超时）。
        """
        if self.connected:
            return

        if self.KIND is None:
            raise RuntimeError("KIND must be set by subclass to 'color' or 'depth'")

        device_info = _find_orbbec_device_info(self.serial_number)
        self.device_name = device_info.get("name")
        self.pipeline = OrbbecPipeline.get_or_create(self.serial_number, self.device_name)
        self.pipeline.connect_client(client_id=id(self), cfg=self.config, kind=self.KIND)

        # warmup：等待当前 camera 视图能读到帧（由 config.warmup_s 决定）
        # 仅等待 frames object != None 仍可能拿不到 color/depth view，因此这里直接用 self.read() 校验。
        if warmup:
            warmup_s = getattr(self.config, "warmup_s", 0) or 5
            start_t = time.time()
            got_frame = False
            last_error: Exception | None = None
            while time.time() - start_t < warmup_s:
                try:
                    if self.read() is not None:
                        got_frame = True
                        break
                except Exception as e:
                    last_error = e
                time.sleep(0.05)  # 避免 warmup busy loop 占满 CPU

            if not got_frame and warmup_s:
                msg = (
                    f"Warmup timed out after {warmup_s}s for {self} "
                    f"(kind={self.KIND}). First frame may arrive later."
                )
                if last_error is not None:
                    logger.warning(f"{msg} Last error during warmup: {last_error}")
                else:
                    logger.warning(msg)
                logger.warning(f"Pipeline status at warmup timeout: {self.pipeline.get_status()}")
            elif got_frame:
                logger.info(
                    f"Warmup succeeded for {self} (kind={self.KIND}) in {time.time() - start_t:.3f}s."
                )

        self.connected = True

    def _read_from_frames(self, frames: Any, color_mode: ColorMode | None = None) -> Optional[np.ndarray]:
        raise NotImplementedError

    def async_read(self, timeout_ms: float = 5000) -> np.ndarray:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                break

            frames, seq = self.pipeline.wait_for_next_frames(self._last_pipeline_seq, timeout_s=remaining_s)
            if frames is None or seq == self._last_pipeline_seq:
                break

            self._last_pipeline_seq = seq
            frame = self._read_from_frames(frames)
            if frame is not None:
                self._last_frame_ts = time.time()
                return frame

        last_age = None if self._last_frame_ts is None else (time.time() - self._last_frame_ts)
        raise TimeoutError(
            f"Timed out waiting for frame from {self} (kind={self.KIND}) after {timeout_ms}ms. "
            f"last_frame_age_s={last_age}, pipeline_status={self.pipeline.get_status()}"
        )

    def disconnect(self) -> None:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} already disconnected.")

        self.connected = False
        self.pipeline.disconnect_client(id(self))
        OrbbecPipeline.drop_if_unused(self.serial_number, self.pipeline)


class OrbbecColorCamera(OrbbecBaseCamera):
    """共享 pipeline 上的彩色逻辑相机视图。"""

    KIND = "color"

    def __init__(self, config: "OrbbecColorCameraConfig"):
        super().__init__(config)
        # 彩色特有
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self._last_missing_color_log_ts: float = 0.0

    def read(self, color_mode: "ColorMode" | None = None) -> Optional[np.ndarray]:
        """
        从 pipeline 的 latest frames 中提取 color view 并做后处理。
        仅读取 pipeline.get_latest_frames()，绝不调用 wait_for_frames()。
        """
        frames = self.pipeline.get_latest_frames()
        if frames is None:
            return None

        return self._read_from_frames(frames, color_mode=color_mode)

    def _read_from_frames(self, frames: Any, color_mode: "ColorMode" | None = None) -> Optional[np.ndarray]:
        if frames is None:
            return None

        color_frame = frames.get_color_frame()
        if color_frame is None:
            now = time.time()
            if now - self._last_missing_color_log_ts > 2.0:
                self._last_missing_color_log_ts = now
                logger.debug(
                    f"{self} missing color_frame (kind={self.KIND}). Pipeline status: {self.pipeline.get_status()}"
                )
            return None

        # 将 SDK 的原始数据变为 numpy array（尽量按期望尺寸解释 buffer）
        target_h = int(getattr(self.config, "height", 0) or 0)
        target_w = int(getattr(self.config, "width", 0) or 0)

        data = color_frame.get_data()
        try:
            buf = np.frombuffer(data, dtype=np.uint8)
            expected = int(self.pipeline.color_height) * int(self.pipeline.color_width) * 3
            if buf.size == expected:
                img = buf.reshape((int(self.pipeline.color_height), int(self.pipeline.color_width), 3))
            else:
                raw = np.asanyarray(data)
                img = raw.reshape((int(self.pipeline.color_height), int(self.pipeline.color_width), 3))
        except Exception as e:
            logger.warning(f"Failed to decode Orbbec color buffer into HxWx3 array: {e}")
            return None

        out = self._postprocess_image(img, color_mode)

        # 系统要求：输出尺寸必须与 json 中配置一致（H/W/3）
        if target_h > 0 and target_w > 0 and (
            out.ndim != 3 or out.shape[0] != target_h or out.shape[1] != target_w
        ):
            out = cv2.resize(out, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        elif out.ndim == 3 and out.shape[2] == 1:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        return out

    def _postprocess_image(self, image: np.ndarray, color_mode: "ColorMode" | None = None) -> np.ndarray:
        """对原始 RGB 图像做颜色顺序和旋转转换。"""
        # 校验 requested color_mode
        if color_mode and color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(f"Invalid requested color mode '{color_mode}'.")

        out = image
        # 按实例配置转换颜色通道顺序（相机配置 color_mode 表示存储格式）
        if getattr(self, "color_mode", None) == ColorMode.BGR:
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        # 旋转处理（若有）
        if self.rotation:
            out = cv2.rotate(out, self.rotation)

        return out


class OrbbecDepthCamera(OrbbecBaseCamera):
    """共享 pipeline 上的深度逻辑相机视图。"""

    KIND = "depth"

    def __init__(self, config: "OrbbecDepthCameraConfig"):
        super().__init__(config)
        self.warmup_s = config.warmup_s
        self._last_missing_depth_log_ts: float = 0.0
        self.depth_alpha = float(config.depth_alpha)

    def read(self, color_mode: "ColorMode" | None = None) -> Optional[np.ndarray]:
        """
        从 pipeline 的 latest frames 中提取 depth view 并做后处理。

        当前输出仍然是可视化友好的伪彩深度图，而不是原始 `uint16 depth map`。
        如果上层以后需要原始深度，应该在这里新增专门路径，而不是在调用侧反向猜测。
        """
        frames = self.pipeline.get_latest_frames()
        if frames is None:
            return None

        return self._read_from_frames(frames)

    def _read_from_frames(self, frames: Any, color_mode: "ColorMode" | None = None) -> Optional[np.ndarray]:
        del color_mode
        if frames is None:
            return None

        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            now = time.time()
            if now - self._last_missing_depth_log_ts > 2.0:
                self._last_missing_depth_log_ts = now
                logger.debug(
                    f"{self} missing depth_frame (kind={self.KIND}). Pipeline status: {self.pipeline.get_status()}"
                )
            return None

        target_h = int(getattr(self.config, "height", 0) or 0)
        target_w = int(getattr(self.config, "width", 0) or 0)

        # SDK depth 原始数据通常是 Y16；优先按 uint16 解码，确保 reshape 与配置一致。
        # 这里一旦 reshape 失败，就说明 pipeline 协商参数和实际到来的 buffer 不一致。
        data = depth_frame.get_data()
        try:
            depth_u16 = np.frombuffer(data, dtype=np.uint16).reshape(
                (int(self.pipeline.depth_height), int(self.pipeline.depth_width))
            )
        except Exception:
            try:
                raw = np.asanyarray(data)
                depth_u16 = raw.reshape((int(self.pipeline.depth_height), int(self.pipeline.depth_width))).astype(
                    np.uint16, copy=False
                )
            except Exception as e:
                logger.warning(f"Failed to decode Orbbec depth buffer into HxW uint16 array: {e}")
                return None

        mask_valid = depth_u16 > 0
        depth_8u = cv2.convertScaleAbs(depth_u16, alpha=self.depth_alpha)
        img = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
        img[~mask_valid] = 0
        # 旋转（若有）
        if self.rotation:
            img = cv2.rotate(img, self.rotation)

        # 系统要求：输出尺寸必须与 json 中配置一致（H/W/3）
        if target_h > 0 and target_w > 0 and (img.shape[0] != target_h or img.shape[1] != target_w):
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        return img


SharedOrbbecManager = OrbbecPipeline
SharedOrbbecColorCamera = OrbbecColorCamera
SharedOrbbecDepthCamera = OrbbecDepthCamera
