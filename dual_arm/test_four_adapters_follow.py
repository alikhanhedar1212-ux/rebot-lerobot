"""Four-adapter dual-pair follow: pair 1 is 1 -> 0, pair 2 is 3 -> 2."""

from __future__ import annotations

import threading
import time

from motorbridge import Controller

from lerobot_robot_seeed_b601 import (
    SeeedB601DMFollower,
    SeeedB601DMFollowerConfig,
    SeeedB601DMLeader,
    SeeedB601DMLeaderConfig,
)

from test_two_adapters_follow import (
    FOLLOWER_DIRECTIONS,
    FOLLOWER_IDS,
    VELOCITIES,
    validate_startup_action,
)


# The DM Device SDK enumeration order does not match USB physical-port order.
# These indices were identified read-only from motor ID banks and joint motion.
PAIR_INDICES = ((1, 0), (3, 2))
# Both calibrated leader grippers open in the negative motor-angle direction,
# matching their follower.  Use raw 1:1 mapping for both pairs.
PAIR_GRIPPER_DIRECTIONS = (1.0, 1.0)
PAIR1_TUNED_VELOCITIES = [58.0, 58.0, 72.0, 48.0, 39.0, 58.0, 90.0]
PAIR1_TUNED_ALPHA = {
    "shoulder_pan.pos": 0.5975,
    "shoulder_lift.pos": 0.5175,
    "elbow_flex.pos": 0.5175,
    "wrist_flex.pos": 0.4775,
    "wrist_yaw.pos": 0.4275,
    "wrist_roll.pos": 0.5375,
    "gripper.pos": 0.75,
}
PAIR1_TUNED_DEADBAND = {
    "shoulder_pan.pos": 0.5025,
    "shoulder_lift.pos": 0.5025,
    "elbow_flex.pos": 0.8525,
    "wrist_flex.pos": 0.4525,
    "wrist_yaw.pos": 0.4525,
    "wrist_roll.pos": 0.8525,
    "gripper.pos": 0.20,
}
PAIR2_TUNED_VELOCITIES = [40.0, 40.0, 50.0, 35.0, 30.0, 40.0, 35.0]
PAIR2_TUNED_ALPHA = {
    "shoulder_pan.pos": 0.35,
    "shoulder_lift.pos": 0.30,
    "elbow_flex.pos": 0.30,
    "wrist_flex.pos": 0.28,
    "wrist_yaw.pos": 0.25,
    "wrist_roll.pos": 0.30,
    "gripper.pos": 0.30,
}
PAIR2_TUNED_DEADBAND = {
    "shoulder_pan.pos": 0.40,
    "shoulder_lift.pos": 0.40,
    "elbow_flex.pos": 0.45,
    "wrist_flex.pos": 0.35,
    "wrist_yaw.pos": 0.35,
    "wrist_roll.pos": 0.45,
    "gripper.pos": 0.40,
}


def profile_filtered(
    action: dict[str, float],
    previous: dict[str, float],
    alpha_by_joint: dict[str, float],
    deadband_by_joint: dict[str, float],
) -> dict[str, float]:
    result = {}
    for key, raw_value in action.items():
        raw = float(raw_value)
        old = previous.get(key)
        if old is None:
            result[key] = raw
        elif abs(raw - old) <= deadband_by_joint.get(key, 0.55):
            result[key] = old
        else:
            alpha = alpha_by_joint.get(key, 0.30)
            result[key] = old + alpha * (raw - old)
    return result


def make_pair(
    pair_number: int,
    leader_index: int,
    follower_index: int,
    velocities: list[float] | None = None,
):
    leader_bus = Controller.from_dm_device(
        "usb2canfd", "0", device_index=leader_index
    )
    try:
        follower_bus = Controller.from_dm_device(
            "usb2canfd", "0", device_index=follower_index
        )
    except Exception:
        leader_bus.close()
        raise

    leader = SeeedB601DMLeader(
        SeeedB601DMLeaderConfig(id=f"master_{pair_number}"), bus=leader_bus
    )
    follower_directions = dict(FOLLOWER_DIRECTIONS)
    follower_directions["gripper"] = PAIR_GRIPPER_DIRECTIONS[pair_number - 1]
    follower = SeeedB601DMFollower(
        SeeedB601DMFollowerConfig(
            id=f"follower_{pair_number}",
            port="",
            can_adapter="dm_device",
            motor_can_ids=FOLLOWER_IDS,
            joint_directions=follower_directions,
            pos_vel_velocity=VELOCITIES if velocities is None else velocities,
        ),
        bus=follower_bus,
    )
    return leader_bus, follower_bus, leader, follower


def safely(callable_, label: str) -> None:
    try:
        callable_()
    except Exception as error:
        print(f"Cleanup warning ({label}): {error}")


def follow_worker(
    pair_index: int,
    leader,
    follower,
    stop_event: threading.Event,
    errors: list[tuple[int, Exception]],
) -> None:
    previous: dict[str, float] = {}
    alpha = PAIR1_TUNED_ALPHA if pair_index == 0 else PAIR2_TUNED_ALPHA
    deadband = PAIR1_TUNED_DEADBAND if pair_index == 0 else PAIR2_TUNED_DEADBAND
    try:
        while not stop_event.is_set():
            action = leader.get_action()
            previous = profile_filtered(action, previous, alpha, deadband)
            follower.send_action(previous)
            time.sleep(0.01)
    except Exception as error:
        errors.append((pair_index + 1, error))
        stop_event.set()


def main() -> None:
    pairs = []
    workers: list[threading.Thread] = []
    stop_event = threading.Event()
    worker_errors: list[tuple[int, Exception]] = []
    try:
        for pair_number, (leader_index, follower_index) in enumerate(
            PAIR_INDICES, start=1
        ):
            velocities = (
                PAIR1_TUNED_VELOCITIES
                if pair_number == 1
                else PAIR2_TUNED_VELOCITIES
            )
            pairs.append(
                make_pair(
                    pair_number,
                    leader_index,
                    follower_index,
                    velocities=velocities,
                )
            )

        for pair_number, (_, _, leader, follower) in enumerate(pairs, start=1):
            print(f"Connecting pair {pair_number}...")
            leader.connect(calibrate=False)
            follower.connect(calibrate=False)
            follower.configure()

        # Validate both leader gripper zeros before enabling either follower.
        # This keeps startup atomic: one bad zero cannot move just one pair.
        for pair_number, (_, _, leader, _) in enumerate(pairs, start=1):
            validate_startup_action(leader.get_action(), f"pair {pair_number} leader")

        # Do not enable either pair until all four arms have connected and both
        # followers have passed configuration.
        for _, _, _, follower in pairs:
            follower.enable_motors()
        for _, _, leader, _ in pairs:
            leader.start_gravity_compensation()

        print("Four-adapter follow ON: 1->0 and 3->2. Ctrl+C to stop.")
        for pair_index, (_, _, leader, follower) in enumerate(pairs):
            worker = threading.Thread(
                target=follow_worker,
                args=(pair_index, leader, follower, stop_event, worker_errors),
                name=f"follow-pair-{pair_index + 1}",
                daemon=True,
            )
            workers.append(worker)
            worker.start()
        while not stop_event.wait(0.2):
            pass
        if worker_errors:
            pair_number, error = worker_errors[0]
            raise RuntimeError(f"pair {pair_number} follow worker failed: {error}")
    except KeyboardInterrupt:
        print("\nStopping four-adapter follow...")
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=2.0)
        # Stop commanded motion first, then release resources in reverse order.
        for pair_number, (_, _, leader, follower) in reversed(
            list(enumerate(pairs, start=1))
        ):
            safely(follower.disable_motors, f"pair {pair_number} follower disable")
            safely(
                leader.stop_gravity_compensation,
                f"pair {pair_number} leader gravity stop",
            )
            safely(
                lambda arm=follower: arm.disconnect(hard=True),
                f"pair {pair_number} follower disconnect",
            )
            safely(leader.disconnect, f"pair {pair_number} leader disconnect")
        for pair_number, (leader_bus, follower_bus, _, _) in reversed(
            list(enumerate(pairs, start=1))
        ):
            safely(follower_bus.close, f"pair {pair_number} follower bus close")
            safely(leader_bus.close, f"pair {pair_number} leader bus close")
        print("Done.")


if __name__ == "__main__":
    main()
