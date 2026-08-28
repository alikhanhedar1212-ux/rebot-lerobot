import logging
import math
import os
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from lerobot.cameras import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import MotorCalibration
from motorbridge import Controller as MotorBridgeController, Mode as MotorBridgeMode
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position


@dataclass
class SeeedB601FollowerConfigBase:
    """Base configuration for the Seeed B601 Follower arm."""

    # Communication port for CAN adapter (e.g., "can0" for SocketCAN, or "/dev/ttyACM0" for Damiao serial bridge)
    port: str
    
    # CAN adapter type:
    #   "socketcan"   - SocketCAN based adapters (PCAN, slcan, embedded can controller, etc.)
    #   "damiao"      - Damiao dedicated serial bridge (legacy, uses from_dm_serial)
    #   "dm_device"   - Damiao DM_Device native (e.g. DM-USB2FDCAN, uses from_dm_device)
    #   "robstride"   - RobStride dedicated adapter (placeholder, not yet supported)
    can_adapter: str = "socketcan"

    # Baud rate for Damiao serial bridge (only used when can_adapter="damiao")
    dm_serial_baud: int = 921600

    # DM_Device parameters (only used when can_adapter="dm_device")
    dm_device_type: str = "usb2canfd"
    dm_device_channel: str = "0"
    # Physical adapter index returned by the DM Device SDK enumerator.
    dm_device_index: int = 0

    disable_torque_on_disconnect: bool = True

    # Standard LeRobot record/eval loops do not call enable_motors().  Opt in
    # when this robot is controlled directly by lerobot-record.
    enable_motors_on_connect: bool = False
    calibrate_on_connect: bool = True

    # The standalone recording/evaluation path should normally stop in place
    # and disable instead of commanding an automatic return-to-zero trajectory.
    safe_zero_on_disconnect: bool = True

    # Max relative target for joint movements, in degrees
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    
    # Motor configuration must be provided by concrete subclasses.
    # Maps motor names to (send_can_id, recv_can_id)
    motor_can_ids: dict[str, tuple[int, int]] = field(default_factory=dict)

    # Control parameters are defined by concrete subclasses so different motor families
    # can keep their own defaults.
    ## Default target velocity for joints running in POS_VEL mode, in degrees/s.
    pos_vel_velocity: float | list[float] = field(default_factory=list)

    ## Default torque/current ration for gripper's FORCE_POS mode, in range [0,1].
    force_pos_torque_ration: float = 0.1

    # Soft joint limits in degrees. Concrete subclasses should define defaults.
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Per-joint action direction/scale applied before joint-limit clipping.
    # Use -1 for sign flip, 1 for no flip, and other values when scaling is required.
    joint_directions: dict[str, float] = field(default_factory=dict)

    # Temperature protection thresholds (degrees Celsius, read from each motor's t_mos).
    # Concrete subclasses may override per motor family.
    temp_alarm_threshold_c: float = 80.0              # print HIGH TEMP warning above this
    temp_overheat_threshold_c: float = 100.0           # raise KeyboardInterrupt above this
    temp_emergency_disable_threshold_c: float = 135.0  # safe_zero emergency disable-torque limit


logger = logging.getLogger(__name__)

# 0.25 kHz adapter command ceiling => at least 4 ms between adjacent frames.
BUS_COMMAND_INTERVAL_S = 0.004


FOLLOWER_GRIPPER_MOTOR = "gripper"
LONG_TIMEOUT_SEC = 0.1
MEDIUM_TIMEOUT_SEC = 0.10

class SeeedB601FollowerBase(Robot):
    """
    Base class for Seeed B601 Follower Arms (DM and RS variants).
    Uses CAN bus communication via motorbridge.
    """

    motor_type: str = ""

    def __init__(self, config: SeeedB601FollowerConfigBase, bus=None):
        super().__init__(config)
        self.config = config
        # ``bus`` may be injected by a shared-bus owner (e.g. the dual-arm server).
        # When injected, this object does NOT own the bus and will not close it.
        self.bus = bus
        self._owns_bus = bus is None
        self._dm_device_prime_bus = None
        self._connected = False
        self.motors = {}
        self.motor_names = list(config.motor_can_ids.keys())
        self._control_modes_configured = False
        self._in_safe_zero = False
        self._emergency_disable_requested = False
        # When True, disconnect() skips safe_zero(). Set by calibrate() so the
        # lerobot-calibrate entrypoint (which calls calibrate() then immediately
        # disconnect()) does not move the arm back to zero. connect() resets this
        # after its internal calibrate() call so normal use still safe-zeros.
        self._skip_safe_zero_on_disconnect = False

        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for observation and action spaces."""
        features: dict[str, type] = {}
        for motor in self.motor_names:
            features[f"{motor}.pos"] = float
            # features[f"{motor}.vel"] = float
            # features[f"{motor}.torque"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features for observation space."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Combined observation features from motors and cameras."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._connected

    def _add_motors_to_bus(self):
        """Must be implemented by subclasses to add specific motor types to self.bus."""
        raise NotImplementedError

    def _create_bus(self):
        """Create the CAN bus according to the configured adapter type.

        Only called when ``bus`` was NOT injected (i.e. this object owns the bus).
        """
        if self.config.can_adapter == "dm_device":
            # The vendor runtime can fail when index 0 is the first adapter
            # opened in a fresh process. Safe ACT evaluation opts into keeping
            # index 1 open (without motors or commands) while index 0 is used.
            prime_index = os.environ.get("REBOT_DM_DEVICE_PRIME_INDEX")
            if prime_index is not None:
                parsed_prime_index = int(prime_index)
                if parsed_prime_index != self.config.dm_device_index:
                    self._dm_device_prime_bus = MotorBridgeController.from_dm_device(
                        dm_device_type=self.config.dm_device_type,
                        dm_channel=self.config.dm_device_channel,
                        device_index=parsed_prime_index,
                    )
            try:
                return MotorBridgeController.from_dm_device(
                    dm_device_type=self.config.dm_device_type,
                    dm_channel=self.config.dm_device_channel,
                    device_index=self.config.dm_device_index,
                )
            except Exception:
                if self._dm_device_prime_bus is not None:
                    self._dm_device_prime_bus.close()
                    self._dm_device_prime_bus = None
                raise
        if self.config.can_adapter == "damiao":
            return MotorBridgeController.from_dm_serial(
                serial_port=self.config.port,
                baud=self.config.dm_serial_baud,
            )
        if self.config.can_adapter == "robstride":
            raise NotImplementedError(
                "RobStride dedicated USB-to-CAN adapter is not yet supported in motorbridge Python SDK."
            )
        # Default: socketcan (PCAN, slcan, etc.)
        return MotorBridgeController(channel=self.config.port)

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the follower arm and optionally calibrate.

        Motors are left DISABLED after connect(); call enable_motors() to enable
        torque (this is the dual-arm server's "enable" command).
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if self.bus is None:
            logger.info(f"Connecting arm on {self.config.port} (adapter={self.config.can_adapter})...")
            self.bus = self._create_bus()
            self._owns_bus = True

        self._add_motors_to_bus()

        if not self.is_calibrated and calibrate and self.config.calibrate_on_connect:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()
            # calibrate() set _skip_safe_zero_on_disconnect for the calibrate
            # script's benefit; clear it here so a later disconnect (after
            # normal use of this connected arm) still runs safe_zero().
            self._skip_safe_zero_on_disconnect = False

        for cam in self.cameras.values():
            cam.connect()

        self._connected = True
        logger.info(f"{self} connected (motors disabled; call enable_motors() to enable).")
        if self.config.enable_motors_on_connect:
            self.enable_motors()
            logger.info(f"{self} motors enabled automatically by configuration.")

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        return bool(self.calibration)

    def calibrate(self) -> None:
        """Calibration procedure for B601."""
        # Mark so that the disconnect() following this (as in the
        # lerobot-calibrate entrypoint) skips safe_zero(). connect() clears
        # this flag after its internal calibrate() call.
        self._skip_safe_zero_on_disconnect = True
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using calibration file associated with the id {self.id}")
                return

        logger.info(f"\nRunning calibration for {self}")

        self.disable_motors()

        print(
            "\nCalibration: Set Zero Position\n"
            "Please MANUALLY move the robot to its ZERO POSITION, and close its gripper.\n"
            "Reference the B601 manual for Zero Pose (generally the default sit-down position).\n"
        )
        input("Press ENTER when ready...")

        for motor in self.motors.values():
            motor.set_zero_position()
            time.sleep(LONG_TIMEOUT_SEC)
        
        logger.info("Arm zero position set.")

        logger.info("Setting range: -90° to +90° by default for all joints")
        self.calibration = {}
        for motor_name, (send_id, recv_id) in self.config.motor_can_ids.items():
            self.calibration[motor_name] = MotorCalibration(
                id=send_id,
                drive_mode=0,
                homing_offset=0,
                range_min=-90,
                range_max=90,
            )

        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def set_zero(self) -> None:
        """Set each motor's current physical pose as its zero reference (no prompt).

        Used by the dual-arm server's ``set_zero`` command; the interactive
        "move to zero + press ENTER" prompt is handled by the client instead.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.disable_motors()
        for motor in self.motors.values():
            motor.set_zero_position()
            time.sleep(LONG_TIMEOUT_SEC)
        logger.info(f"{self} zero position set.")

    def configure(self) -> None:
        """Set each motor's control mode. Motors stay disabled; call enable_motors() to enable."""
        num_retry = 9
        target_modes = {}

        # Validate/set every control mode before sending PID register writes.
        # Mixing those writes with CTRL_MODE reads can leave unrelated ACK
        # frames queued in the DM-USB2FDCAN runtime.
        for motor_name, motor in self.motors.items():
            target_mode = MotorBridgeMode.MIT if self.motor_type == "rs" else (
                MotorBridgeMode.FORCE_POS
                if motor_name == FOLLOWER_GRIPPER_MOTOR
                else MotorBridgeMode.POS_VEL
            )
            target_modes[motor_name] = target_mode

            # DM motors often answer register reads reliably but do not send the
            # write acknowledgement expected by ensure_mode when the requested
            # mode is already active.  Avoid that unnecessary write: register 10
            # is CTRL_MODE.  If the read fails or reports another mode, retain
            # the normal ensure_mode path so a real mismatch is never ignored.
            mode_is_current = False
            if self.motor_type == "dm":
                read_error = None
                for read_attempt in range(3):
                    try:
                        read_mode = getattr(
                            motor,
                            "damiao_get_param_u32",
                            motor.get_register_u32,
                        )
                        current_mode = read_mode(10, 2000)
                        mode_is_current = current_mode == int(target_mode)
                        logger.info(
                            "%s control mode readback=%s target=%s",
                            motor_name,
                            current_mode,
                            int(target_mode),
                        )
                        break
                    except Exception as e:
                        read_error = e
                        time.sleep(MEDIUM_TIMEOUT_SEC)
                else:
                    raise RuntimeError(
                        f"{motor_name} control mode register 10 read failed after 3 attempts: "
                        f"{read_error}"
                    ) from read_error

            if not mode_is_current:
                for attempt in range(num_retry + 1):
                    try:
                        motor.ensure_mode(target_mode, 1000)  # 1000ms timeout
                        break
                    except Exception as e:
                        if attempt == num_retry:
                            raise RuntimeError(
                                f"{motor_name} ensure_mode {int(target_mode)} failed after "
                                f"{num_retry + 1} attempts: {e}"
                            ) from e
                        time.sleep(MEDIUM_TIMEOUT_SEC)
            logger.info(f"{motor_name} ensure mode {target_mode}")
            time.sleep(MEDIUM_TIMEOUT_SEC)

        # Write PID parameters only after all mode reads/changes are complete.
        for motor_name, motor in self.motors.items():
            if target_modes[motor_name] == MotorBridgeMode.POS_VEL and self.motor_type == "dm":
                try:
                    motor.write_register_f32(25, 0.0125)  # KP_ASR (vel_kp)
                    time.sleep(BUS_COMMAND_INTERVAL_S)
                    motor.write_register_f32(26, 0.004)   # KI_ASR (vel_ki)
                    time.sleep(BUS_COMMAND_INTERVAL_S)
                    motor.write_register_f32(27, 60.0)    # KP_APR (pos_kp)
                    time.sleep(BUS_COMMAND_INTERVAL_S)
                    motor.write_register_f32(28, 0.5)     # KI_APR (pos_ki)
                    time.sleep(BUS_COMMAND_INTERVAL_S)
                except Exception:
                    logger.debug(f"{motor_name} PID param write skipped", exc_info=True)

        self._control_modes_configured = True

    def _refresh_feedback(self, timeout_s: float = 1.0) -> None:
        """Request and poll feedback until all own motors respond (or timeout).

        A single request+poll round-trip is not enough for the DM-USB2FDCAN: the
        CAN response needs time to arrive. Loop with a short sleep and re-request
        any motor that has not reported yet (same pattern as Seeed's web control).
        """
        for motor in self.motors.values():
            motor.request_feedback()
        deadline = time.perf_counter() + timeout_s
        received: set[str] = set()
        while time.perf_counter() < deadline:
            time.sleep(0.03)
            try:
                self.bus.poll_feedback_once()
            except Exception:
                logger.warning("poll feedback failed.")
            for name, motor in self.motors.items():
                if name not in received and motor.get_state() is not None:
                    received.add(name)
            if len(received) == len(self.motors):
                break
            for name, motor in self.motors.items():
                if name not in received:
                    motor.request_feedback()

    def enable_motors(self) -> None:
        """Enable follower torque and hold the current pose.

        Reads each joint's current position, sets control modes, sends a "hold
        current position" command BEFORE enabling (so POS_VEL does not chase a
        stale target), then enables torque. Per-motor only -- never bus-level.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 1. Read current positions (motor space, radians).
        self._refresh_feedback()
        current_rad: dict[str, float] = {}
        for motor_name, motor in self.motors.items():
            state = motor.get_state()
            current_rad[motor_name] = state.pos if state is not None else 0.0

        # 2. Set control modes while motors are still disabled. Runtime owners
        # may preconfigure them and mark this flag to avoid redundant writes.
        if not self._control_modes_configured:
            self.configure()

        # 3. Send "hold current position" BEFORE enabling.
        for idx, motor_name in enumerate(self.motors):
            motor = self.motors[motor_name]
            pos_rad = current_rad[motor_name]
            vel_deg_s = (
                self.config.pos_vel_velocity[idx]
                if isinstance(self.config.pos_vel_velocity, list)
                else self.config.pos_vel_velocity
            )
            vel_rad = math.radians(vel_deg_s)
            if motor_name == FOLLOWER_GRIPPER_MOTOR:
                if self.motor_type == "rs":
                    motor.send_mit(pos_rad, 0, 0, 1.5, 0.0)
                else:
                    motor.send_force_pos(pos_rad, vel_rad, self.config.force_pos_torque_ration)
            else:
                if self.motor_type == "rs":
                    kp = getattr(self.config, "mit_kp", {}).get(motor_name, 0.0)
                    kd = getattr(self.config, "mit_kd", {}).get(motor_name, 0.0)
                    motor.send_mit(pos_rad, 0, kp, kd, 0)
                else:
                    motor.send_pos_vel(pos_rad, vel_rad)
            time.sleep(BUS_COMMAND_INTERVAL_S)

        # 4. Enable torque once per motor. Repeating enable waves can
        # re-initialize an already-running DM motor and make joints 2/3 stop.
        for motor in self.motors.values():
            motor.enable()
            time.sleep(BUS_COMMAND_INTERVAL_S)
        logger.info(f"{self} follower motors enabled (holding current pose).")

    def disable_motors(self) -> None:
        """Disable follower motor torque (arm goes limp). Per-motor, NOT bus-level."""
        for motor in self.motors.values():
            motor.disable()
        logger.info(f"{self} follower motors disabled.")

    def disable_torque(self) -> None:
        """Disable follower motor torque so the arm can be moved by hand during read-only debugging."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.disable_motors()

    def _read_motor_temperatures(self, refresh_feedback: bool = True) -> dict[str, float]:
        """Return per-motor MOS temperatures from fresh or cached feedback.

        ``send_action`` is latency-sensitive and uses the latest states already
        maintained by the shared bus.  A synchronous seven-motor refresh can
        wait for its one-second timeout and must not sit in the control path.
        Explicit diagnostics retain the fresh-feedback default.
        """
        if refresh_feedback:
            self._refresh_feedback()

        temps: dict[str, float] = {}
        for motor_name, motor in self.motors.items():
            state = motor.get_state()
            if state is not None:
                temps[motor_name] = state.t_mos

        return temps

    def _check_motor_temperatures(
        self,
        alarm_threshold_c: float,
        overheat_threshold_c: float,
        context: str = "",
        refresh_feedback: bool = True,
    ) -> dict[str, float]:
        """Read motor MOS temperatures once, print a HIGH TEMP warning for any
        motor above ``alarm_threshold_c``, and raise ``KeyboardInterrupt`` if any
        motor exceeds ``overheat_threshold_c`` (aborts the control loop).

        Returns the dict of ``{motor_name: t_mos_c}`` that was read.
        """
        temperatures = self._read_motor_temperatures(refresh_feedback=refresh_feedback)
        label = f" in {context}" if context else ""
        for motor_name, temp_c in temperatures.items():
            if temp_c > alarm_threshold_c:
                print(
                    f"[HIGH TEMP] {motor_name} t_mos={temp_c:.2f}C > {alarm_threshold_c:.2f}C"
                )
            if temp_c > overheat_threshold_c:
                logger.error(
                    "Overheat detected%s: %s t_mos=%.2fC > %.2fC.",
                    label,
                    motor_name,
                    temp_c,
                    overheat_threshold_c,
                )
                raise KeyboardInterrupt("Overheat detected")
        return temperatures

    def mit_output_torque_limit(
        self,
        motor: Any,
        pos_target_rad: float,
    ) -> float | None:
        """Compute MIT torque command from target position and motor state."""
        return 0.0

    def safe_zero(
        self,
        step_interval_s: float = 0.02,
        exit_on_complete: bool = True,
        velocity_scale: float = 1.0,
    ) -> None:
        """Move arm joints back to zero in a safer two-stage interpolation.

        Stage 1: shoulder_pan / wrist_flex / wrist_yaw / wrist_roll -> 0
        Stage 2: shoulder_lift / elbow_flex -> 0
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._in_safe_zero:
            logger.warning("safe_zero skipped: already running.")
            return

        if step_interval_s < 0.0:
            raise ValueError("step_interval_s must be >= 0")
        if velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be > 0")

        self._in_safe_zero = True
        original_pos_vel_velocity = self.config.pos_vel_velocity
        if isinstance(original_pos_vel_velocity, list):
            self.config.pos_vel_velocity = [
                float(velocity) * velocity_scale for velocity in original_pos_vel_velocity
            ]
        else:
            self.config.pos_vel_velocity = float(original_pos_vel_velocity) * velocity_scale
        logger.info(
            "safe_zero velocity scale: %.2fx (%s -> %s)",
            velocity_scale,
            original_pos_vel_velocity,
            self.config.pos_vel_velocity,
        )
        try:
            stage_1 = [
                n for n in ("shoulder_pan", "wrist_flex", "wrist_yaw", "wrist_roll")
                if n in self.motor_names
            ]
            stage_2 = [
                n for n in ("shoulder_lift", "elbow_flex")
                if n in self.motor_names
            ]
            controlled_joints = stage_1 + [name for name in stage_2 if name not in stage_1]

            if not controlled_joints:
                logger.warning("safe_zero skipped: no arm joints found.")
                return

            def _read_action_pos(joint_name: str) -> float:
                motor = self.motors.get(joint_name)
                if motor is None:
                    raise RuntimeError(f"safe_zero failed: motor '{joint_name}' not found")

                max_retry = 10
                for attempt in range(1, max_retry + 1):
                    try:
                        motor.request_feedback()
                        self.bus.poll_feedback_once()
                    except Exception:
                        logger.debug(
                            "safe_zero feedback poll failed for %s (attempt %d/%d)",
                            joint_name,
                            attempt,
                            max_retry,
                        )

                    state = motor.get_state()
                    if state is not None:
                        current_deg = math.degrees(state.pos)
                        direction = self.config.joint_directions.get(joint_name, 1.0) or 1.0
                        return current_deg / direction

                    if attempt < max_retry:
                        time.sleep(MEDIUM_TIMEOUT_SEC)

                raise RuntimeError(
                    f"safe_zero failed: unable to read state for '{joint_name}' after {max_retry} attempts"
                )

            def _frame_count(
                starts: dict[str, float],
                targets: dict[str, float] | None = None,
            ) -> int:
                targets = targets or {}
                max_delta_deg = max(
                    (abs(targets.get(j, 0.0) - v) for j, v in starts.items()),
                    default=0.0,
                )
                return max(1, math.ceil(max_delta_deg * 2.0))

            def _interp_to_zero(
                active_starts: dict[str, float],
                hold_joints: dict[str, float],
                targets: dict[str, float] | None = None,
            ) -> bool:
                if not active_starts:
                    return False
                targets = targets or {}
                frames = _frame_count(active_starts, targets)
                emergency_disable_threshold_c = self.config.temp_emergency_disable_threshold_c
                for frame in range(1, frames + 1):
                    temperatures = self._read_motor_temperatures()
                    for motor_name, temp_c in temperatures.items():
                        if temp_c > emergency_disable_threshold_c:
                            logger.error(
                                "Auto-disable on overtemperature during safe_zero: %s t_mos=%.2fC > %.2fC.",
                                motor_name,
                                temp_c,
                                emergency_disable_threshold_c,
                            )
                            self._emergency_disable_requested = True
                            self.disable_torque()
                            logger.error("safe_zero aborted: emergency overtemperature.")
                            return True

                    ratio = frame / frames
                    action: RobotAction = {}
                    for joint, start in hold_joints.items():
                        action[f"{joint}.pos"] = start
                    for joint, start in active_starts.items():
                        target = targets.get(joint, 0.0)
                        action[f"{joint}.pos"] = start + (target - start) * ratio
                    self.send_action(action)
                    if step_interval_s > 0.0:
                        time.sleep(step_interval_s)

                return False

            stage_1_start = {joint: _read_action_pos(joint) for joint in stage_1}
            stage_2_start = {joint: _read_action_pos(joint) for joint in stage_2}

            logger.info("safe_zero stage1 start: joints=%s", stage_1)
            if _interp_to_zero(stage_1_start, stage_2_start):
                return

            # Stage 2: move shoulder_lift/elbow_flex back to zero, and bring the gripper back to
            # 170° if it is currently past 180° (avoids leaving it wide open).
            # NOTE: gripper action-space sign differs per variant (RS: +, DM: -),
            # so we compare magnitude and preserve the current sign for the target.
            stage_2_active = dict(stage_2_start)
            stage_2_targets: dict[str, float] = {}
            if FOLLOWER_GRIPPER_MOTOR in self.motors:
                try:
                    gripper_pos = _read_action_pos(FOLLOWER_GRIPPER_MOTOR)
                except RuntimeError as e:
                    logger.warning("safe_zero: could not read gripper position: %s", e)
                    gripper_pos = None
                if gripper_pos is not None and abs(gripper_pos) > 180.0:
                    gripper_target = math.copysign(170.0, gripper_pos)
                    logger.info(
                        "safe_zero gripper: %.2f° (abs > 180°), returning to %.2f°",
                        gripper_pos,
                        gripper_target,
                    )
                    stage_2_active[FOLLOWER_GRIPPER_MOTOR] = gripper_pos
                    stage_2_targets[FOLLOWER_GRIPPER_MOTOR] = gripper_target

            logger.info("safe_zero stage2 start: joints=%s", stage_2)
            if _interp_to_zero(
                stage_2_active,
                {joint: 0.0 for joint in stage_1},
                stage_2_targets,
            ):
                return
            logger.info("safe_zero done.")
            time.sleep(2.0)
            if exit_on_complete:
                # Raise KeyboardInterrupt so upper-level control loops handle this
                # the same way as Ctrl+C.
                raise KeyboardInterrupt("safe_zero completed")
        finally:
            self.config.pos_vel_velocity = original_pos_vel_velocity
            self._in_safe_zero = False

    def get_observation(self) -> RobotObservation:
        """Get current observation from robot."""
        start = time.perf_counter()

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict: dict[str, Any] = {}

        # Request and poll feedback from motorbridge
        for motor in self.motors.values():
            motor.request_feedback()
        try:
            self.bus.poll_feedback_once()
        except:
            logger.warning(f"can bus poll feedback failed.")

        for motor_name, motor in self.motors.items():
            state = motor.get_state()
            if state is not None:
                # motorbridge works natively in radians. Convert to degrees.
                obs_dict[f"{motor_name}.pos"] = math.degrees(state.pos)
                obs_dict[f"{motor_name}.vel"] = math.degrees(state.vel)
                obs_dict[f"{motor_name}.torque"] = state.torq
            else:
                obs_dict[f"{motor_name}.pos"] = 0.0
                obs_dict[f"{motor_name}.vel"] = 0.0
                obs_dict[f"{motor_name}.torque"] = 0.0

        # Capture images
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} get_observation took: {dt_ms:.1f}ms")
        # logger.debug(f"Observation: {obs_dict}")

        return obs_dict

    def send_action(
        self,
        action: RobotAction
    ) -> RobotAction:
        """Send action command to robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self._in_safe_zero:
            self._check_motor_temperatures(
                self.config.temp_alarm_threshold_c,
                self.config.temp_overheat_threshold_c,
                context="send_action",
                refresh_feedback=False,
            )

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Apply per-joint direction/scale mapping before clipping.
        # Default to identity (1.0): a joint missing from joint_directions must
        # NOT be silently commanded to 0.
        for motor_name, position in goal_pos.items():
            direction = self.config.joint_directions.get(motor_name, 1.0)
            position = position * direction
            # print(f"motor_name: {motor_name}, position: {position}")
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                clipped_position = max(min_limit, min(max_limit, position))
                if clipped_position != position:
                    logger.debug(f"Clipped {motor_name} from {position:.2f} to {clipped_position:.2f}")
                position = clipped_position

            goal_pos[motor_name] = position

        # To tolerate 6-DOF leader arms that don't have a wrist_yaw joint, we can allow the follower to ignore missing wrist_yaw commands by treating them as 0.
        if 'wrist_yaw' not in goal_pos:
            goal_pos['wrist_yaw'] = 0.0

        # Safety: Cap relative target
        if self.config.max_relative_target is not None:
            # We need current position in degrees to compare against relative limit safely
            present_pos = {}
            for motor_name, motor in self.motors.items():
                state = motor.get_state()
                if state is not None:
                    present_pos[motor_name] = math.degrees(state.pos)
                else:
                    present_pos[motor_name] = 0.0
            
            goal_present_pos = {key: (g_pos, present_pos.get(key, g_pos)) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Prepare and send commands
        for motor_name, position_degrees in goal_pos.items():
            try:
                idx = self.motor_names.index(motor_name)
            except ValueError:
                idx = 0 # Fallback

            # Convert target position from degrees to radians for motorbridge
            pos_rad = math.radians(position_degrees)
            vel_deg_s = (
                self.config.pos_vel_velocity[idx]
                if isinstance(self.config.pos_vel_velocity, list)
                else self.config.pos_vel_velocity
            )
            vel_rad = math.radians(vel_deg_s)

            motor = self.motors.get(motor_name)
            if motor is not None:
                if motor_name == FOLLOWER_GRIPPER_MOTOR:
                    if self.motor_type == "rs":
                        tau_ff = self.mit_output_torque_limit(motor, pos_rad)
                        if tau_ff is None:
                            tau_ff = 0.0
                        motor.send_mit(0, 0, 0, 1.5, tau_ff)
                        logger.debug(
                            f"Sent MIT command to {motor_name}: pos={position_degrees:.2f}°, "
                            f"tau_ff={tau_ff:.2f}"
                        )
                    else:
                        motor.send_force_pos(pos_rad, vel_rad, self.config.force_pos_torque_ration)
                        logger.debug(f"Sent FORCE_POS command to {motor_name}: pos={position_degrees:.2f}°, vel={vel_deg_s:.2f}°/s, ratio={0.1}")
                else:
                    if self.motor_type == "rs":
                        kp = getattr(self.config, "mit_kp", {}).get(motor_name, 0.0)
                        kd = getattr(self.config, "mit_kd", {}).get(motor_name, 0.0)
                        motor.send_mit(pos_rad, 0, kp, kd, 0)
                        logger.debug(
                            f"Sent MIT command to {motor_name}: "
                            f"pos={position_degrees:.2f}°, kp={kp}, kd={kd}"
                        )
                    else:
                        motor.send_pos_vel(pos_rad, vel_rad)
                        logger.debug(f"Sent POS_VEL command to {motor_name}: target={pos_rad:.2f},pos={position_degrees:.2f}°, vel={vel_deg_s:.2f}°/s")
                time.sleep(BUS_COMMAND_INTERVAL_S)

        # motorbridge sends packets mostly synchronously here over loop, 
        # so we don't need a bulk send command through ctypes.

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self, hard: bool = False):
        """Disconnect from robot.

        ``hard=True`` skips safe_zero (used by the dual-arm server for an
        immediate stop on Ctrl+C). The bus is only closed when this object owns
        it (``_owns_bus``); an injected shared bus is left open for its owner.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if hard:
            logger.info("hard disconnect: skipping safe_zero.")
        elif not self._owns_bus:
            # Shared bus: the bus owner (dual-arm server) controls shutdown; this
            # object must not move the arm on its way out. Fall through to the
            # per-motor disable below.
            logger.info("shared bus: skipping safe_zero (bus owned by server).")
        elif not self.config.safe_zero_on_disconnect:
            logger.info("safe_zero disabled by configuration.")
        elif (
            not self._in_safe_zero
            and not self._emergency_disable_requested
            and not self._skip_safe_zero_on_disconnect
        ):
            try:
                self.safe_zero(exit_on_complete=False)
            except Exception:
                logger.exception("safe_zero during disconnect failed.")
        elif self._skip_safe_zero_on_disconnect:
            logger.info(
                "safe_zero skipped on disconnect: calibrate context "
                "(_skip_safe_zero_on_disconnect=True)."
            )

        for motor in self.motors.values():
            if self.config.disable_torque_on_disconnect:
                motor.disable()
            motor.clear_error()
            motor.close()
        self.motors = {}

        if self._owns_bus and self.bus is not None:
            self.bus.close()
        if self._dm_device_prime_bus is not None:
            self._dm_device_prime_bus.close()
            self._dm_device_prime_bus = None
        self.bus = None
        self._connected = False

        for cam in self.cameras.values():
            cam.disconnect()

        self._emergency_disable_requested = False
        logger.info(f"{self} disconnected.")
