"""Two-adapter master/follower test: leader index 1, follower index 0."""

from __future__ import annotations

import os
import time

from motorbridge import Controller

from lerobot_robot_seeed_b601 import (
    SeeedB601DMFollower,
    SeeedB601DMFollowerConfig,
    SeeedB601DMLeader,
    SeeedB601DMLeaderConfig,
)


FOLLOWER_IDS = {
    "shoulder_pan": (0x21, 0x31),
    "shoulder_lift": (0x22, 0x32),
    "elbow_flex": (0x23, 0x33),
    "wrist_flex": (0x24, 0x34),
    "wrist_yaw": (0x25, 0x35),
    "wrist_roll": (0x26, 0x36),
    "gripper": (0x27, 0x37),
}

FOLLOWER_DIRECTIONS = {
    "shoulder_pan": 1.0,
    "shoulder_lift": 1.0,
    "elbow_flex": 1.0,
    "wrist_flex": 1.0,
    "wrist_yaw": 1.0,
    "wrist_roll": 1.0,
    # Pair 1 was verified on hardware to use the same raw gripper direction.
    # Negating the leader's negative opening angle would produce a positive
    # target, which is clipped to zero by the follower's [-270, 0] limit.
    "gripper": 1.0,
}

# A correctly zeroed leader gripper should be near zero at startup.  Refuse to
# enable a follower when the leader still reports an old/raw half-turn offset.
MAX_STARTUP_GRIPPER_OFFSET_DEG = 90.0

VELOCITIES = [60.0, 60.0, 75.0, 50.0, 40.0, 60.0, 90.0]
ALPHA = {
    "shoulder_pan.pos": 0.675,
    "shoulder_lift.pos": 0.575,
    "elbow_flex.pos": 0.575,
    "wrist_flex.pos": 0.525,
    "wrist_yaw.pos": 0.475,
    "gripper.pos": 0.75,
}
DEADBAND = {
    "shoulder_pan.pos": 0.25,
    "shoulder_lift.pos": 0.25,
    "wrist_flex.pos": 0.20,
    "wrist_yaw.pos": 0.20,
    "gripper.pos": 0.20,
}


def filtered(action: dict[str, float], previous: dict[str, float]) -> dict[str, float]:
    result = {}
    for key, raw_value in action.items():
        raw = float(raw_value)
        old = previous.get(key)
        alpha = ALPHA.get(key, 0.60)
        deadband = DEADBAND.get(key, 0.60)
        if old is None or abs(raw - old) > deadband:
            result[key] = raw if old is None else old + alpha * (raw - old)
        else:
            result[key] = old
    return result


def validate_startup_action(action: dict[str, float], label: str = "leader") -> None:
    gripper = action.get("gripper.pos")
    if gripper is None:
        raise RuntimeError(f"{label} startup preflight failed: missing gripper feedback")
    if abs(float(gripper)) > MAX_STARTUP_GRIPPER_OFFSET_DEG:
        raise RuntimeError(
            f"{label} startup preflight failed: gripper={float(gripper):.1f} deg; "
            "re-zero the leader gripper before enabling the follower"
        )


def main() -> None:
    leader_index = int(os.environ.get("REBOT_LEADER_INDEX", "1"))
    follower_index = int(os.environ.get("REBOT_FOLLOWER_INDEX", "0"))
    leader_bus = Controller.from_dm_device("usb2canfd", "0", device_index=leader_index)
    follower_bus = Controller.from_dm_device("usb2canfd", "0", device_index=follower_index)
    leader = SeeedB601DMLeader(SeeedB601DMLeaderConfig(id="master"), bus=leader_bus)
    follower = SeeedB601DMFollower(
        SeeedB601DMFollowerConfig(
            id="follower",
            port="",
            can_adapter="dm_device",
            motor_can_ids=FOLLOWER_IDS,
            joint_directions=FOLLOWER_DIRECTIONS,
            pos_vel_velocity=VELOCITIES,
        ),
        bus=follower_bus,
    )

    try:
        leader.connect(calibrate=False)
        follower.connect(calibrate=False)
        follower.configure()
        validate_startup_action(leader.get_action())
        follower.enable_motors()
        leader.start_gravity_compensation()
        print(f"Two-adapter follow ON: leader=index{leader_index}, follower=index{follower_index}. Ctrl+C to stop.")
        previous = {}
        while True:
            action = leader.get_action()
            previous = filtered(action, previous)
            follower.send_action(previous)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopping two-adapter follow...")
    finally:
        follower.disable_motors()
        leader.stop_gravity_compensation()
        follower.disconnect(hard=True)
        leader.disconnect()
        follower_bus.close()
        leader_bus.close()
        print("Done.")


if __name__ == "__main__":
    main()
