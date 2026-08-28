from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("seeed_b601_dm_leader")
@dataclass
class SeeedB601DMLeaderConfig(TeleoperatorConfig):
    """Configuration for a Seeed B601-DM arm used as a hand-moved LEADER (read-only)."""

    # Damiao USB-CAN-FD device type and channel. For the DM-USB2FDCAN this is
    # type "usb2canfd", channel "0".
    dm_device_type: str = "usb2canfd"
    dm_device_channel: str = "0"
    # Physical adapter index returned by the DM Device SDK enumerator.
    dm_device_index: int = 0

    # Standard LeRobot teleoperate/record loops only call connect(),
    # get_action(), and disconnect().  Enable this for a B601 leader that must
    # enter gravity compensation automatically while it is used as a teleop.
    auto_gravity_compensation: bool = False
    calibrate_on_connect: bool = True
    max_startup_gripper_offset_deg: float | None = None

    # Optional filtering applied before get_action() returns.  Keeping the
    # filter in the teleoperator means lerobot-record stores exactly the same
    # action that it sends to the follower.
    action_filter_alpha: dict[str, float] = field(default_factory=dict)
    action_filter_deadband: dict[str, float] = field(default_factory=dict)

    # Maps motor names to (send_can_id, recv_can_id). These are the MASTER arm's
    # default IDs (0x01~0x07 send, 0x11~0x17 feedback).
    motor_can_ids: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "shoulder_pan": (0x01, 0x11),
            "shoulder_lift": (0x02, 0x12),
            "elbow_flex": (0x03, 0x13),
            "wrist_flex": (0x04, 0x14),
            "wrist_yaw": (0x05, 0x15),
            "wrist_roll": (0x06, 0x16),
            "gripper": (0x07, 0x17),
        }
    )
