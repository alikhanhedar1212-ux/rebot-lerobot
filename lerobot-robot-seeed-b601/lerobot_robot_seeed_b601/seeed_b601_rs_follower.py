import logging

from .config_seeed_b601_rs_follower import SeeedB601RSFollowerConfig
from .seeed_b601_follower import SeeedB601FollowerBase

logger = logging.getLogger(__name__)


class SeeedB601RSFollower(SeeedB601FollowerBase):
    """
    Seeed B601-RS Robot Arm (RobStride Motors).
    Uses CAN bus communication via motorbridge SDK.
    """

    config_class = SeeedB601RSFollowerConfig
    name = "seeed_b601_rs_follower"
    motor_type = "rs"

    motor_model_mapping = {
        "shoulder_pan":  "rs-06",
        "shoulder_lift": "rs-06",
        "elbow_flex":    "rs-06",
        "wrist_flex":    "rs-00",
        "wrist_yaw":     "rs-00",
        "wrist_roll":    "rs-00",
        "gripper":       "rs-00",
    }

    def _add_motors_to_bus(self):
        for motor_name, (send_id, recv_id) in self.config.motor_can_ids.items():
            motor_type_str = self.motor_model_mapping[motor_name]
            self.motors[motor_name] = self.bus.add_robstride_motor(send_id, recv_id, motor_type_str)

    def mit_output_torque_limit(
        self,
        motor,
        pos_target: float,
    ) -> float | None:
        if motor is None:
            return None
        self.bus.poll_feedback_once()
        state = motor.get_state()
        if state is None:
            return None

        # Impedance control: tau = K*(x_r - x) + B*(x_dot_r - x_dot)
        control_dt_s = 0.02
        # Assumed control frequency.
        if control_dt_s <= 0.0:
            return None

        if not hasattr(self, "_gripper_prev_target_pos"):
            self._gripper_prev_target_pos = None
        if not hasattr(self, "_gripper_prev_filtered_target_vel"):
            self._gripper_prev_filtered_target_vel = None
        if not hasattr(self, "_gripper_prev_state_pos"):
            self._gripper_prev_state_pos = None

        prev_target_pos = self._gripper_prev_target_pos
        if prev_target_pos is None:
            target_vel = 0.0
        else:
            target_vel = (pos_target - prev_target_pos) / control_dt_s
        self._gripper_prev_target_pos = pos_target

        lpf_alpha = 0.3
        target_vel_max = 3.0
        prev_filtered_vel = self._gripper_prev_filtered_target_vel
        if prev_filtered_vel is None:
            filtered_target_vel = target_vel
        else:
            filtered_target_vel = (
                lpf_alpha * target_vel + (1.0 - lpf_alpha) * prev_filtered_vel
            )
        target_vel = max(-target_vel_max, min(target_vel_max, filtered_target_vel))
        self._gripper_prev_filtered_target_vel = target_vel

        prev_state_pos = self._gripper_prev_state_pos
        if prev_state_pos is None:
            estimated_state_vel = 0.0
        else:
            estimated_state_vel = (state.pos - prev_state_pos) / control_dt_s
        self._gripper_prev_state_pos = state.pos

        kp = float(self.config.gripper_mit_kp)
        kd = float(self.config.gripper_mit_kd)
        impedance_torque = (
            kp * (pos_target - state.pos)
            + kd * (target_vel - state.vel)
        )
        logger.debug(
            "Gripper MIT terms: pos_target=%.4f rad, state_pos=%.4f rad, "
            "target_vel=%.4f rad/s, state_vel=%.4f rad/s",
            pos_target,
            state.pos,
            target_vel,
            state.vel,
        )

        # Use dedicated grasp/hold torque limit when estimated motor speed is small.
        max_torque = (
            self.config.gripper_mit_torque_limit
            if abs(estimated_state_vel) > 0.25
            else self.config.gripper_mit_hold_torque_limit
        )
        motor.request_feedback()
        return max(-max_torque, min(max_torque, impedance_torque))
