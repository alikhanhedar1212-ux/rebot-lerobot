#!/usr/bin/env python3
"""Standalone check: enable/disable the FOLLOWER (slave) arm alone.

Usage:
    python dual_arm/test_follower_enable.py

Connects to the DM-USB2FDCAN, adds the follower arm's 7 motors (send IDs
0x21~0x27 / feedback IDs 0x31~0x37), enables torque while holding the current
pose, then disables on Ctrl+C. Verifies the follower arm can enable/disable.

Requires ONLY the follower arm to be physically connected and powered.
"""
import argparse
import time

from motorbridge import Controller as MotorBridgeController

from lerobot_robot_seeed_b601 import SeeedB601DMFollower, SeeedB601DMFollowerConfig


FOLLOWER_IDS = {
    "shoulder_pan": (0x21, 0x31),
    "shoulder_lift": (0x22, 0x32),
    "elbow_flex": (0x23, 0x33),
    "wrist_flex": (0x24, 0x34),
    "wrist_yaw": (0x25, 0x35),
    "wrist_roll": (0x26, 0x36),
    "gripper": (0x27, 0x37),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    config = SeeedB601DMFollowerConfig(
        id="follower",
        port="",  # unused: bus is injected below
        can_adapter="dm_device",
        motor_can_ids=FOLLOWER_IDS,
    )

    bus = MotorBridgeController.from_dm_device(
        "usb2canfd",
        "0",
        device_index=args.device_index,
    )
    follower = SeeedB601DMFollower(config, bus=bus)

    follower.connect(calibrate=False)
    print("Follower connected (disabled). Enabling torque and holding current pose...\n")
    follower.enable_motors()
    # Read a genuinely post-enable frame; get_state() may otherwise still
    # expose the cached status from before enable().
    for motor in follower.motors.values():
        motor.request_feedback()
        time.sleep(0.01)
    time.sleep(0.10)
    for _ in range(3):
        bus.poll_feedback_once()
        time.sleep(0.03)
    for name, motor in follower.motors.items():
        state = motor.get_state()
        status = state.status_code if state is not None else None
        print(f"  {name}: status={status}")
    print("Follower ENABLED (holding current pose). Ctrl+C to disable and exit.\n")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nDisabling follower...")
    finally:
        follower.disable_motors()
        follower.disconnect(hard=True)
        bus.close()
        print("Done.")


if __name__ == "__main__":
    main()
