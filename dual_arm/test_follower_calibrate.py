#!/usr/bin/env python3
"""Set the follower arm zero pose through DM device index 1."""

import argparse

from motorbridge import Controller

from lerobot_robot_seeed_b601 import SeeedB601DMFollower, SeeedB601DMFollowerConfig

from test_two_adapters_follow import FOLLOWER_DIRECTIONS, FOLLOWER_IDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Set a follower arm's current pose as zero")
    parser.add_argument("--device-index", type=int, default=1)
    args = parser.parse_args()

    bus = Controller.from_dm_device("usb2canfd", "0", device_index=args.device_index)
    follower = SeeedB601DMFollower(
        SeeedB601DMFollowerConfig(
            id=f"follower_{args.device_index}",
            port="",
            can_adapter="dm_device",
            motor_can_ids=FOLLOWER_IDS,
            joint_directions=FOLLOWER_DIRECTIONS,
        ),
        bus=bus,
    )

    try:
        follower.connect(calibrate=False)
        print("Move the follower arm to its ZERO pose, then press ENTER...")
        input()
        follower.set_zero()
        print("Follower zero position set.")
    finally:
        if follower.is_connected:
            follower.disconnect(hard=True)
        bus.close()
        print("Done.")


if __name__ == "__main__":
    main()
