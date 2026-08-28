import logging
import math
import time
from typing import Any

import numpy as np
from lerobot.motors import MotorCalibration
from lerobot.processor import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from motorbridge import Controller as MotorBridgeController, Mode as MotorBridgeMode

from .config_seeed_b601_dm_leader import SeeedB601DMLeaderConfig
from .gravity_compensation import compute_gravity_torque

logger = logging.getLogger(__name__)


# The 6 arm joints, in the same order the gravity model (and thus
# compute_gravity_torque) expects. The gripper is NOT gravity-compensated.
ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
)

# Gravity-compensation MIT gains. kp is 0 (pure feed-forward): the position
# reference tracks the current q, so a non-zero kp degenerates to latency damping
# that hinders motion. kd damps only the pitch joints to suppress oscillation.
GRAVITY_KP = 0.0
# Per-joint velocity damping. Joints with ~0 gravity torque (shoulder_pan,
# wrist_yaw, wrist_roll) get 0 so they stay free; the pitch joints get a little
# damping to suppress oscillation.
GRAVITY_KD = np.array([0.0, 0.5, 0.5, 0.5, 0.0, 0.0], dtype=np.float64)
# Per-joint torque scale. With the correct DevArm masses the model needs no
# per-joint fudge, so this stays all 1.0 (the ROS2 DM config has no tau_scale).
GRAVITY_TAU_SCALE = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

# Gravity-compensation direction sign. -1.0 => send tau_g as-is (the DM motor's
# MIT tau sign is opposite to the model's). The old code used |gain| = 2.5 as a
# fudge for the underestimated B601 masses; with the DevArm masses that is gone.
GRAVITY_GAIN = -1.0

# Gripper free-drive MIT velocity gain. 0.0 => pure zero-torque (freest the DM
# firmware allows). A small NEGATIVE value (e.g. -0.05..-0.2) applies velocity-
# proportional friction compensation: the motor nudges in the direction you move
# it, helping overcome gearbox/cogging friction. Too negative => the gripper
# creeps on its own. Tune in small steps.
GRIPPER_FREE_KD = 0.0

# 0.25 kHz adapter command ceiling => at least 4 ms between adjacent frames.
BUS_COMMAND_INTERVAL_S = 0.004


class SeeedB601DMLeader(Teleoperator):
    """Seeed B601-DM arm used as a hand-moved LEADER teleoperator.

    Motors are disabled by default and only their feedback is read. For teleop the
    leader can also be switched into gravity-compensation mode (``start_gravity_compensation``),
    which torque-enables the 6 arm joints in MIT; ``get_action()`` then keeps feeding
    the gravity torque while still returning each joint's read position.
    """

    config_class = SeeedB601DMLeaderConfig
    name = "seeed_b601_dm_leader"

    motor_model_mapping = {
        "shoulder_pan": "dm4340p",
        "shoulder_lift": "dm4340p",
        "elbow_flex": "dm4340p",
        "wrist_flex": "dm4310",
        "wrist_yaw": "dm4310",
        "wrist_roll": "dm4310",
        "gripper": "dm4310",
    }

    def __init__(self, config: SeeedB601DMLeaderConfig, bus=None):
        super().__init__(config)
        self.config = config
        # ``bus`` may be injected by a shared-bus owner (e.g. the dual-arm server).
        self.bus = bus
        self._owns_bus = bus is None
        self._connected = False
        self.motors = {}
        self.motor_names = list(config.motor_can_ids.keys())
        self._gc_active = False
        self._gc_q = None
        self._filtered_action: dict[str, float] = {}

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motor_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration) and set(self.calibration) == set(self.motor_names)

    def _add_motors_to_bus(self):
        for motor_name, (send_id, recv_id) in self.config.motor_can_ids.items():
            motor_str = self.motor_model_mapping[motor_name].upper().replace("DM", "")
            self.motors[motor_name] = self.bus.add_damiao_motor(send_id, recv_id, motor_str)

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the leader arm. Motors are left disabled (read-only)."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if self.bus is None:
            self.bus = MotorBridgeController.from_dm_device(
                self.config.dm_device_type,
                self.config.dm_device_channel,
                device_index=self.config.dm_device_index,
            )
            self._owns_bus = True

        self._add_motors_to_bus()

        # Leader is pure read-only: disable every motor so the arm can be moved
        # by hand and only its feedback is read.
        for motor in self.motors.values():
            motor.disable()
            time.sleep(BUS_COMMAND_INTERVAL_S)

        if not self.is_calibrated and calibrate and self.config.calibrate_on_connect:
            self.calibrate()

        self._connected = True
        logger.info(f"{self} connected (read-only, motors disabled).")
        self._filtered_action.clear()
        if self.config.max_startup_gripper_offset_deg is not None:
            startup_action = self.get_action()
            gripper = float(startup_action["gripper.pos"])
            limit = float(self.config.max_startup_gripper_offset_deg)
            if abs(gripper) > limit:
                raise RuntimeError(
                    f"leader startup preflight failed: gripper={gripper:.1f} deg; "
                    f"expected abs(gripper) <= {limit:.1f} deg"
                )
        if self.config.auto_gravity_compensation:
            self.start_gravity_compensation()

    def configure(self) -> None:
        """Read-only leader: nothing to enable; keep motors disabled."""
        for motor in self.motors.values():
            motor.disable()

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using calibration file associated with the id {self.id}")
                return

        logger.info(f"\nRunning calibration for {self}")
        input(
            "\nCalibration: move the master arm to its ZERO pose, then press ENTER..."
        )
        for motor in self.motors.values():
            motor.set_zero_position()

        self.calibration = {}
        for motor_name, (send_id, _recv_id) in self.config.motor_can_ids.items():
            self.calibration[motor_name] = MotorCalibration(
                id=send_id,
                drive_mode=0,
                homing_offset=0,
                range_min=-90,
                range_max=90,
            )
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def set_zero(self) -> None:
        """Set each motor's current pose as zero (no prompt). Used by the server's set_zero command."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        # Zeroing while gravity compensation is active would leave ``_gc_active``
        # true with the arm joints re-disabled; stop it first for consistency.
        self.stop_gravity_compensation()
        for motor in self.motors.values():
            motor.disable()
        for motor in self.motors.values():
            motor.set_zero_position()
        logger.info(f"{self} zero position set.")

    def _refresh_feedback(
        self,
        timeout_s: float = 1.0,
        motor_names: tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, Any]:
        """Request and poll feedback until all leader motors respond (or timeout).

        A single request+poll round-trip is not enough for the DM-USB2FDCAN: the
        CAN response needs time to arrive. Re-request missing motors until the
        deadline. ``get_state`` is read exactly once per poll and the captured
        state is reused by the caller.
        """
        requested_names = list(self.motors) if motor_names is None else list(motor_names)
        requested_motors = {name: self.motors[name] for name in requested_names}
        for motor in requested_motors.values():
            motor.request_feedback()
            time.sleep(BUS_COMMAND_INTERVAL_S)
        deadline = time.perf_counter() + timeout_s
        states: dict[str, Any] = {}
        while time.perf_counter() < deadline:
            time.sleep(0.03)
            try:
                self.bus.poll_feedback_once()
            except Exception:
                logger.warning("poll feedback failed.")
            for name, motor in requested_motors.items():
                if name not in states:
                    state = motor.get_state()
                    if state is not None:
                        states[name] = state
            if len(states) == len(requested_motors):
                break
            for name, motor in requested_motors.items():
                if name not in states:
                    motor.request_feedback()
                    time.sleep(BUS_COMMAND_INTERVAL_S)
        return states

    def get_action(self, allow_missing_gripper: bool = False) -> RobotAction:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        requested_names = list(ARM_JOINT_NAMES) if allow_missing_gripper else list(self.motors)
        states = self._refresh_feedback(motor_names=requested_names)

        action: dict[str, Any] = {}
        for motor_name in requested_names:
            state = states.get(motor_name)
            if state is not None:
                action[f"{motor_name}.pos"] = math.degrees(state.pos)
            else:
                logger.warning(f"leader motor {motor_name} has no feedback.")
        missing = [name for name in requested_names if name not in states]
        blocking_missing = [
            name for name in missing if not (allow_missing_gripper and name == "gripper")
        ]
        if blocking_missing:
            # Never substitute 0 degrees for a missing joint: during following
            # that could command a sudden move on the follower.
            raise RuntimeError(
                "leader feedback incomplete: "
                + ", ".join(blocking_missing)
                + " (check power/CAN connection)"
            )

        # If gravity compensation is active, fold the MIT torque send into the same
        # feedback poll so the leader is gravity-compensated AND its position is
        # read for teleop in a single round-trip. Done AFTER the feedback check so a
        # total feedback loss raises before any wrong MIT torque is commanded.
        if self._gc_active:
            self._send_gravity_mit(self._read_arm_positions_rad(states), states)

        if self.config.action_filter_alpha or self.config.action_filter_deadband:
            filtered: dict[str, float] = {}
            for key, raw_value in action.items():
                raw = float(raw_value)
                old = self._filtered_action.get(key)
                deadband = float(self.config.action_filter_deadband.get(key, 0.0))
                alpha = float(self.config.action_filter_alpha.get(key, 1.0))
                if not 0.0 < alpha <= 1.0:
                    raise ValueError(f"action_filter_alpha[{key!r}] must be in (0, 1]")
                if old is None:
                    filtered[key] = raw
                elif abs(raw - old) <= deadband:
                    filtered[key] = old
                else:
                    filtered[key] = old + alpha * (raw - old)
            self._filtered_action = filtered
            return dict(filtered)

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        raise NotImplementedError("Feedback is not implemented for the B601-DM leader.")

    # ------------------------------------------------------------ gravity compensation
    def _read_arm_positions_rad(self, states: dict[str, Any] | None = None) -> np.ndarray:
        """Read the 6 arm joint positions in radians (model order) from the
        last-polled feedback. Call ``_refresh_feedback()`` first for fresh data."""
        q = np.zeros(len(ARM_JOINT_NAMES), dtype=np.float64)
        for i, name in enumerate(ARM_JOINT_NAMES):
            state = states.get(name) if states is not None else self.motors[name].get_state()
            if state is not None:
                q[i] = float(state.pos)
        return q

    def start_gravity_compensation(self, skip_mode_config: bool = False) -> None:
        """Switch the leader into gravity-compensation mode.

        Puts the 6 arm motors into MIT mode with soft gains and enables torque,
        then the caller must drive ``gravity_tick()`` in a loop to keep sending
        the gravity feed-forward torque. The arm then feels "weightless".
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self._gc_active:
            return

        def ensure_mit_with_retry(name: str) -> None:
            """Tolerate a dropped register ACK while the USB adapter starts."""
            for attempt in range(1, 4):
                try:
                    self.motors[name].ensure_mode(MotorBridgeMode.MIT, 1000)
                    return
                except Exception as error:  # noqa: BLE001
                    # DM motors can apply the register write even when its ACK is
                    # dropped.  Verify the actual register before retrying/failing.
                    try:
                        actual_mode = int(self.motors[name].get_register_u32(10, 1000))
                    except Exception:  # noqa: BLE001
                        actual_mode = None
                    if actual_mode == int(MotorBridgeMode.MIT):
                        logger.warning(
                            "MIT ACK missing for %s, but register 10 confirms MIT; continuing",
                            name,
                        )
                        return
                    if attempt == 3:
                        raise error
                    logger.warning("MIT mode ACK timeout for %s; retrying (%d/3)", name, attempt)
                    time.sleep(0.1)
        # Mirror the ROS2 start sequence: enable -> read q -> mode_mit -> disable -> enable.
        # The disable -> enable cycle after the mode change is what makes the DM motor
        # actually re-initialize in the new MIT mode.
        for name in ARM_JOINT_NAMES:
            self.motors[name].enable()
        states = self._refresh_feedback()
        q = self._read_arm_positions_rad(states)
        if not skip_mode_config:
            for name in ARM_JOINT_NAMES:
                ensure_mit_with_retry(name)
                time.sleep(0.05)
        for name in ARM_JOINT_NAMES:
            self.motors[name].disable()
        time.sleep(0.1)
        for name in ARM_JOINT_NAMES:
            self.motors[name].enable()
        # Gripper free-drive: switch it to MIT zero-stiffness (kp=kd=tau=0) and
        # ENABLE it. A disabled gripper is stiff because the high-reduction
        # mechanism has to be back-driven through the gearbox + motor cogging;
        # enabling + zero-torque MIT is the free state (ROS2 "free-drive" example).
        gripper_free_drive = True
        if not skip_mode_config:
            try:
                ensure_mit_with_retry("gripper")
            except Exception as error:  # noqa: BLE001
                gripper_free_drive = False
                logger.error(
                    "Gripper MIT mode could not be confirmed; leaving gripper disabled: %s",
                    error,
                )
        self.motors["gripper"].disable()
        time.sleep(0.05)
        if gripper_free_drive:
            self.motors["gripper"].enable()
        self._gc_active = True
        self._gc_q = q
        # Send the first MIT command right away so the motors are never left
        # enabled in MIT mode without a command (avoids an undefined brief state).
        self.gravity_tick()
        logger.info(f"{self} gravity compensation started (kp={GRAVITY_KP}, kd={GRAVITY_KD}).")
        # Debug: verify mode (10, MIT=1) and enable status (0=disabled, 1=enabled).
        self._refresh_feedback()
        for name in ARM_JOINT_NAMES:
            try:
                mode = int(self.motors[name].get_register_u32(10, 500))
            except Exception as e:  # noqa: BLE001
                mode = f"err:{e}"
            state = self.motors[name].get_state()
            status = state.status_code if state is not None else "None"
            print(f"[gc] {name}: mode={mode} status={status} (0=dis,1=en)", flush=True)

    def _send_gravity_mit(self, q: np.ndarray, states: dict[str, Any] | None = None) -> None:
        """Send one MIT command: gravity feed-forward torque at the current q.

        Assumes fresh feedback has already been polled (the caller decides when to
        poll, so read-for-teleop and gravity-comp can share a single round-trip).
        """
        tau_g = compute_gravity_torque(q)            # N*m, 6 elements
        # NOTE: sign flipped so the motor applies the torque that COUNTERACTS
        # gravity (the DM motor's MIT torque sign is opposite to the model's).
        tau_motor = GRAVITY_GAIN * (-tau_g * GRAVITY_TAU_SCALE)
        for i, name in enumerate(ARM_JOINT_NAMES):
            self.motors[name].send_mit(
                float(q[i]),
                0.0,
                GRAVITY_KP,
                float(GRAVITY_KD[i]),
                float(tau_motor[i]),
            )
            time.sleep(BUS_COMMAND_INTERVAL_S)
        self._send_gripper_free(states.get("gripper") if states is not None else None)
        self._gc_q = q
        # Throttled debug output (every ~2 s): commanded torque vs reported torque.
        now = time.perf_counter()
        if now - getattr(self, "_gc_dbg_ts", 0.0) >= 2.0:
            self._gc_dbg_ts = now
            report_states = states or {}
            torq_rep = [
                round(float(report_states[name].torq), 3) if name in report_states else None
                for name in ARM_JOINT_NAMES
            ]
            print(
                f"[gc] q_deg={[round(math.degrees(x), 1) for x in q]} "
                f"tau_cmd={[round(x, 3) for x in tau_motor]} "
                f"torq_rep={torq_rep}",
                flush=True,
            )

    def _send_gripper_free(self, state: Any | None = None) -> None:
        """Send a zero-stiffness MIT command to the gripper (kp=tau=0) so it
        stays freely back-drivable. Uses the last-polled gripper position.
        GRIPPER_FREE_KD adds optional friction compensation (see its comment)."""
        gripper = self.motors["gripper"]
        if state is None:
            state = gripper.get_state()
        q = float(state.pos) if state is not None else 0.0
        gripper.send_mit(q, 0.0, 0.0, GRIPPER_FREE_KD, 0.0)

    def gravity_tick(self) -> None:
        """One gravity-compensation step: poll q, compute tau_g(q), send MIT."""
        if not self._gc_active:
            return
        states = self._refresh_feedback()
        self._send_gravity_mit(self._read_arm_positions_rad(states), states)

    def stop_gravity_compensation(self) -> None:
        """Disable the gravity-compensation torque (arm goes limp again)."""
        if not self._gc_active:
            return
        for name in ARM_JOINT_NAMES:
            self.motors[name].disable()
        self.motors["gripper"].disable()
        self._gc_active = False
        self._gc_q = None
        logger.info(f"{self} gravity compensation stopped.")

    def disconnect(self, hard: bool = False) -> None:
        # ``hard`` is accepted for symmetry with the follower; the leader never
        # runs safe_zero, so it has no effect.
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.stop_gravity_compensation()
        for motor in self.motors.values():
            motor.disable()
            motor.clear_error()
            motor.close()
        self.motors = {}
        self._gc_active = False
        self._gc_q = None
        self._filtered_action.clear()

        if self._owns_bus and self.bus is not None:
            self.bus.close()
        self.bus = None
        self._connected = False
        logger.info(f"{self} disconnected.")
