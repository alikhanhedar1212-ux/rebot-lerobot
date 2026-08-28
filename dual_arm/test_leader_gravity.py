#!/usr/bin/env python3
"""Standalone gravity-compensation test for the master (leader) arm.

Usage:
    python dual_arm/test_leader_gravity.py
    python dual_arm/test_leader_gravity.py --device-index 2

Switches the master arm into gravity compensation (MIT mode + gravity feed-forward
torque) so it feels weightless while you move it by hand. Ctrl+C to stop.

Requires ONLY the master arm to be connected and powered. For the compensation to
feel right, the arm should first be calibrated (set_zero) so its joint angles match
the URDF neutral pose.
"""
import argparse
import time

from motorbridge import Controller as MotorBridgeController

from lerobot_robot_seeed_b601 import SeeedB601DMLeader, SeeedB601DMLeaderConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Test leader-arm gravity compensation")
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()

    config = SeeedB601DMLeaderConfig(id=f"master_{args.device_index}")
    bus = MotorBridgeController.from_dm_device(
        "usb2canfd", "0", device_index=args.device_index
    )
    leader = SeeedB601DMLeader(config, bus=bus)

    leader.connect(calibrate=False)
    leader.start_gravity_compensation()
    print("Gravity compensation ON. Move the master arm; it should feel weightless.")
    print("Ctrl+C to stop.\n")

    tick_s = 0.005  # ~200 Hz compensation loop
    last_print = 0.0
    try:
        while True:
            leader.gravity_tick()
            now = time.perf_counter()
            if now - last_print >= 2.0:
                last_print = now
                print("[gravity comp] running...")
            time.sleep(tick_s)
    except KeyboardInterrupt:
        print("\n\nStopping gravity compensation...")
    finally:
        leader.stop_gravity_compensation()
        leader.disconnect(hard=True)
        bus.close()
        print("Done.")


if __name__ == "__main__":
    main()
