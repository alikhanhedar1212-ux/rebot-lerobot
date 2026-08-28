#!/usr/bin/env python3
"""Standalone check: read the MASTER (leader) arm alone.

Usage:
    python dual_arm/test_leader_read.py

Connects to the DM-USB2FDCAN, adds the master arm's 7 motors (send IDs
0x01~0x07 / feedback IDs 0x11~0x17), disables them (read-only) and prints the
7 joint positions at ~4 Hz. Verifies the master arm can be read.

Requires ONLY the master arm to be physically connected and powered.
"""
import time

from motorbridge import Controller as MotorBridgeController

from lerobot_robot_seeed_b601 import SeeedB601DMLeader, SeeedB601DMLeaderConfig


def main() -> None:
    config = SeeedB601DMLeaderConfig(id="master")

    bus = MotorBridgeController.from_dm_device("usb2canfd", "0")
    leader = SeeedB601DMLeader(config, bus=bus)

    leader.connect(calibrate=False)
    print("Leader connected (read-only). Reading master arm joints... Ctrl+C to exit.\n")

    print_interval_s = 0.5  # 2 Hz terminal print (new line per update, readable)
    last_print = 0.0
    try:
        while True:
            action = leader.get_action()
            now = time.perf_counter()
            if now - last_print >= print_interval_s:
                last_print = now
                line = "  ".join(
                    f"{k.replace('.pos', ''):<12}: {v:8.2f}°" for k, v in action.items()
                )
                print(line, flush=True)
    except KeyboardInterrupt:
        print("\n\nDisconnecting...")
    finally:
        leader.disconnect()
        bus.close()
        print("Done.")


if __name__ == "__main__":
    main()
