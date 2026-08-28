#!/usr/bin/env python3
"""Calibrate (set zero) the master (leader) arm.

Usage:
    python dual_arm/test_leader_calibrate.py
    python dual_arm/test_leader_calibrate.py --device-index 2

Move the master arm to its ZERO pose (the URDF neutral pose used by the gravity
compensation, same pose ROS2 uses), then press ENTER to record each motor's
current position as zero.

Requires ONLY the master arm to be connected and powered.
"""
import argparse

from motorbridge import Controller as MotorBridgeController

from lerobot_robot_seeed_b601 import SeeedB601DMLeader, SeeedB601DMLeaderConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Set a leader arm's current pose as zero")
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()

    config = SeeedB601DMLeaderConfig(id=f"master_{args.device_index}")
    bus = MotorBridgeController.from_dm_device(
        "usb2canfd", "0", device_index=args.device_index
    )
    leader = SeeedB601DMLeader(config, bus=bus)

    leader.connect(calibrate=False)
    print("Move the master arm to its ZERO pose, then press ENTER...")
    input()
    leader.set_zero()
    print("Zero position set.")

    leader.disconnect(hard=True)
    bus.close()
    print("Done.")


if __name__ == "__main__":
    main()
