import time

from .config_seeed_b601_dm_follower import SeeedB601DMFollowerConfig
from .seeed_b601_follower import SeeedB601FollowerBase


HANDLE_REGISTRATION_INTERVAL_S = 0.10


class SeeedB601DMFollower(SeeedB601FollowerBase):
    """
    Seeed B601-DM Robot Arm (6-DOF + Gripper) using Damiao motors.
    Uses CAN bus communication via motorbridge SDK.
    """

    config_class = SeeedB601DMFollowerConfig
    name = "seeed_b601_dm_follower"
    motor_type = "dm"

    motor_model_mapping = {
        "shoulder_pan":  "dm4340p",
        "shoulder_lift": "dm4340p",
        "elbow_flex":    "dm4340p",
        "wrist_flex":    "dm4310",
        "wrist_yaw":     "dm4310",
        "wrist_roll":    "dm4310",
        "gripper":       "dm4310",
    }

    def _add_motors_to_bus(self):
        for motor_name, (send_id, recv_id) in self.config.motor_can_ids.items():
            motor_type_str = self.motor_model_mapping[motor_name]
            motor_str = motor_type_str.upper().replace("DM", "")
            self.motors[motor_name] = self.bus.add_damiao_motor(send_id, recv_id, motor_str)
            # The DM Device backend needs time to publish each new feedback route
            # before another handle is registered on a 14-motor shared channel.
            time.sleep(HANDLE_REGISTRATION_INTERVAL_S)
