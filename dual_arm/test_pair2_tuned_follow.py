"""Conservative tuning run for pair 2: leader index 3 -> follower index 2."""

from __future__ import annotations

import time

from test_four_adapters_follow import make_pair, safely
from test_two_adapters_follow import validate_startup_action


# First tuning pass: lower command speed and stronger smoothing than the
# production two-pair profile.  Values are deliberately conservative.
TUNED_VELOCITIES = [40.0, 40.0, 50.0, 35.0, 30.0, 40.0, 35.0]
TUNED_ALPHA = {
    "shoulder_pan.pos": 0.35,
    "shoulder_lift.pos": 0.30,
    "elbow_flex.pos": 0.30,
    "wrist_flex.pos": 0.28,
    "wrist_yaw.pos": 0.25,
    "wrist_roll.pos": 0.30,
    "gripper.pos": 0.30,
}
TUNED_DEADBAND = {
    "shoulder_pan.pos": 0.40,
    "shoulder_lift.pos": 0.40,
    "elbow_flex.pos": 0.45,
    "wrist_flex.pos": 0.35,
    "wrist_yaw.pos": 0.35,
    "wrist_roll.pos": 0.45,
    "gripper.pos": 0.40,
}


def tuned_filter(action: dict[str, float], previous: dict[str, float]) -> dict[str, float]:
    result = {}
    for key, raw_value in action.items():
        raw = float(raw_value)
        old = previous.get(key)
        if old is None:
            result[key] = raw
            continue
        if abs(raw - old) <= TUNED_DEADBAND.get(key, 0.45):
            result[key] = old
            continue
        alpha = TUNED_ALPHA.get(key, 0.30)
        result[key] = old + alpha * (raw - old)
    return result


def main() -> None:
    leader_bus = follower_bus = leader = follower = None
    try:
        leader_bus, follower_bus, leader, follower = make_pair(
            2, 3, 2, velocities=TUNED_VELOCITIES
        )
        print("Connecting tuned pair 2 (3->2)...")
        leader.connect(calibrate=False)
        follower.connect(calibrate=False)
        follower.configure()
        validate_startup_action(leader.get_action(), "pair 2 leader")
        follower.enable_motors()
        leader.start_gravity_compensation()
        print("Tuned pair 2 follow ON: 3->2. Ctrl+C to stop.")

        previous: dict[str, float] = {}
        while True:
            previous = tuned_filter(leader.get_action(), previous)
            follower.send_action(previous)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopping tuned pair 2 follow...")
    finally:
        if follower is not None:
            safely(follower.disable_motors, "pair 2 follower disable")
        if leader is not None:
            safely(leader.stop_gravity_compensation, "pair 2 leader gravity stop")
        if follower is not None:
            safely(lambda: follower.disconnect(hard=True), "pair 2 follower disconnect")
        if leader is not None:
            safely(leader.disconnect, "pair 2 leader disconnect")
        if follower_bus is not None:
            safely(follower_bus.close, "pair 2 follower bus close")
        if leader_bus is not None:
            safely(leader_bus.close, "pair 2 leader bus close")
        print("Done.")


if __name__ == "__main__":
    main()
