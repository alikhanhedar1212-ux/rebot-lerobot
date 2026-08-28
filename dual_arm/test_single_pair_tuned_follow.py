"""Run one tuned B601 pair in its own process for DM runtime isolation."""

from __future__ import annotations

import argparse
import time

from test_four_adapters_follow import (
    PAIR1_TUNED_ALPHA,
    PAIR1_TUNED_DEADBAND,
    PAIR1_TUNED_VELOCITIES,
    PAIR2_TUNED_ALPHA,
    PAIR2_TUNED_DEADBAND,
    PAIR2_TUNED_VELOCITIES,
    PAIR_INDICES,
    make_pair,
    profile_filtered,
    safely,
)
from test_two_adapters_follow import validate_startup_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    pair_number = args.pair
    leader_index, follower_index = PAIR_INDICES[pair_number - 1]
    if pair_number == 1:
        velocities = PAIR1_TUNED_VELOCITIES
        alpha = PAIR1_TUNED_ALPHA
        deadband = PAIR1_TUNED_DEADBAND
    else:
        velocities = PAIR2_TUNED_VELOCITIES
        alpha = PAIR2_TUNED_ALPHA
        deadband = PAIR2_TUNED_DEADBAND

    leader_bus = follower_bus = leader = follower = None
    try:
        leader_bus, follower_bus, leader, follower = make_pair(
            pair_number, leader_index, follower_index, velocities=velocities
        )
        print(f"[pair {pair_number}] connecting {leader_index}->{follower_index}...", flush=True)
        leader.connect(calibrate=False)
        follower.connect(calibrate=False)
        follower.configure()
        validate_startup_action(leader.get_action(), f"pair {pair_number} leader")
        follower.enable_motors()
        leader.start_gravity_compensation()
        print(f"[pair {pair_number}] follow ON", flush=True)

        previous: dict[str, float] = {}
        while True:
            action = leader.get_action()
            previous = profile_filtered(action, previous, alpha, deadband)
            follower.send_action(previous)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print(f"\n[pair {pair_number}] stopping...", flush=True)
    finally:
        if follower is not None:
            safely(follower.disable_motors, f"pair {pair_number} follower disable")
        if leader is not None:
            safely(leader.stop_gravity_compensation, f"pair {pair_number} leader gravity stop")
        if follower is not None:
            safely(lambda: follower.disconnect(hard=True), f"pair {pair_number} follower disconnect")
        if leader is not None:
            safely(leader.disconnect, f"pair {pair_number} leader disconnect")
        if follower_bus is not None:
            safely(follower_bus.close, f"pair {pair_number} follower bus close")
        if leader_bus is not None:
            safely(leader_bus.close, f"pair {pair_number} leader bus close")
        print(f"[pair {pair_number}] done", flush=True)


if __name__ == "__main__":
    main()
