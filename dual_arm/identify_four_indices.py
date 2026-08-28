"""Read-only identification of arm type and joint positions for DM device indices."""

from __future__ import annotations

import math
import time

from motorbridge import Controller


MOTORS = (
    ("shoulder_pan", "4340P"),
    ("shoulder_lift", "4340P"),
    ("elbow_flex", "4340P"),
    ("wrist_flex", "4310"),
    ("wrist_yaw", "4310"),
    ("wrist_roll", "4310"),
    ("gripper", "4310"),
)


def add_bank(controller: Controller, send_base: int, feedback_base: int):
    return {
        name: controller.add_damiao_motor(send_base + offset, feedback_base + offset, model)
        for offset, (name, model) in enumerate(MOTORS, start=1)
    }


def collect(controller: Controller, motors: dict, timeout_s: float = 3.0):
    states = {}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and len(states) < len(motors):
        for name, motor in motors.items():
            if name not in states:
                motor.request_feedback()
                time.sleep(0.004)
        time.sleep(0.02)
        controller.poll_feedback_once()
        for name, motor in motors.items():
            if name not in states:
                state = motor.get_state()
                if state is not None:
                    states[name] = state
    return states


def positions(states: dict) -> str:
    return ", ".join(
        f"{name}={math.degrees(states[name].pos):.1f}"
        for name, _ in MOTORS
        if name in states
    )


def main() -> None:
    for device_index in range(4):
        results = {}
        for role, send_base, feedback_base in (
            ("leader", 0x00, 0x10),
            ("follower", 0x20, 0x30),
        ):
            controller = Controller.from_dm_device(
                "usb2canfd", "0", device_index=device_index
            )
            try:
                motors = add_bank(controller, send_base, feedback_base)
                results[role] = collect(controller, motors)
            finally:
                controller.close()

        leader_states = results["leader"]
        follower_states = results["follower"]
        if len(leader_states) >= len(follower_states) and leader_states:
            print(
                f"index={device_index} role=leader {len(leader_states)}/7 "
                f"{positions(leader_states)}"
            )
        elif follower_states:
            print(
                f"index={device_index} role=follower {len(follower_states)}/7 "
                f"{positions(follower_states)}"
            )
        else:
            print(f"index={device_index} role=unknown leader=0/7 follower=0/7")


if __name__ == "__main__":
    main()
