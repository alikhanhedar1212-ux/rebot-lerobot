"""Read-only routing test: pair 1 leader index 1, follower index 0."""

from __future__ import annotations

import math
import os
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


def add_arm(controller: Controller, send_base: int, feedback_base: int):
    return {
        name: controller.add_damiao_motor(send_base + i, feedback_base + i, model)
        for i, (name, model) in enumerate(MOTORS)
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


def print_states(label: str, states: dict):
    print(f"{label}: {len(states)}/7 feedback")
    for name, _model in MOTORS:
        state = states.get(name)
        if state is None:
            print(f"  {name:<14} MISSING")
        else:
            print(f"  {name:<14} {math.degrees(state.pos):8.2f} deg")


def main() -> None:
    leader_index = int(os.environ.get("REBOT_LEADER_INDEX", "1"))
    follower_index = int(os.environ.get("REBOT_FOLLOWER_INDEX", "0"))
    leader_bus = Controller.from_dm_device("usb2canfd", "0", device_index=leader_index)
    follower_bus = Controller.from_dm_device("usb2canfd", "0", device_index=follower_index)
    try:
        leader = add_arm(leader_bus, 0x01, 0x11)
        follower = add_arm(follower_bus, 0x21, 0x31)
        leader_states = collect(leader_bus, leader)
        follower_states = collect(follower_bus, follower)
        print_states(f"leader  index={leader_index}", leader_states)
        print_states(f"follower index={follower_index}", follower_states)
        if len(leader_states) != 7 or len(follower_states) != 7:
            raise RuntimeError("incomplete feedback; check adapter-to-arm mapping and CAN wiring")
        print("two-adapter read-only routing test: PASS")
    finally:
        follower_bus.close()
        leader_bus.close()


if __name__ == "__main__":
    main()
