#!/usr/bin/env python3
"""Mock-based correctness test of the master-slave following code.

Runs the leader and follower against a fake ``motorbridge`` bus / motor, so the
core read, gravity-compensation, mapping and lifecycle behavior can be checked
with NO hardware connected. Usage:

    python dual_arm/test_mock_follow.py
"""
import math
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------- fake motorbridge
class Mode:
    MIT = 1
    POS_VEL = 2
    VEL = 3
    FORCE_POS = 4
    ROBSTRIDE_POS_VEL_CSP = 5


class FakeState:
    def __init__(self, pos=0.0, vel=0.0, torq=0.0, status_code=0, t_mos=30.0):
        self.pos = pos
        self.vel = vel
        self.torq = torq
        self.status_code = status_code
        self.t_mos = t_mos


class FakeMotor:
    def __init__(self, send_id, recv_id, model):
        self.send_id, self.recv_id, self.model = send_id, recv_id, model
        self.pos = 0.0
        self.vel = 0.0
        self.torq = 0.0
        self._enabled = False
        self._mode = None
        self.last_mit = None
        self.last_pos_vel = None
        self.last_force_pos = None

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def set_zero_position(self):
        self.pos = 0.0

    def clear_error(self):
        pass

    def close(self):
        pass

    def request_feedback(self):
        pass

    def get_state(self):
        return FakeState(self.pos, self.vel, self.torq, 1 if self._enabled else 0)

    def ensure_mode(self, mode, timeout_ms=1000):
        self._mode = mode

    def send_mit(self, pos, vel, kp, kd, tau):
        self.last_mit = (pos, vel, kp, kd, tau)
        self.torq = tau

    def send_pos_vel(self, pos, vlim):
        self.last_pos_vel = (pos, vlim)
        self.pos = pos

    def send_force_pos(self, pos, vel, ratio):
        self.last_force_pos = (pos, vel, ratio)
        self.pos = pos

    def get_register_u32(self, reg, timeout):
        return 1  # MIT mode

    def write_register_f32(self, reg, val):
        pass


class FakeBus:
    def __init__(self):
        self.motors = []

    def add_damiao_motor(self, send_id, recv_id, model):
        m = FakeMotor(send_id, recv_id, model)
        self.motors.append(m)
        return m

    def poll_feedback_once(self):
        pass

    def close(self):
        pass


class Controller:
    @staticmethod
    def from_dm_device(dm_device_type, dm_device_channel):
        return FakeBus()


fake = types.ModuleType("motorbridge")
fake.Controller = Controller
fake.Mode = Mode
sys.modules["motorbridge"] = fake

# ---------------------------------------------------------------- real classes
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot_robot_seeed_b601 import (  # noqa: E402
    SeeedB601DMLeader,
    SeeedB601DMLeaderConfig,
    SeeedB601DMFollower,
    SeeedB601DMFollowerConfig,
)

FOLLOWER_IDS = {
    "shoulder_pan": (0x21, 0x31),
    "shoulder_lift": (0x22, 0x32),
    "elbow_flex": (0x23, 0x33),
    "wrist_flex": (0x24, 0x34),
    "wrist_yaw": (0x25, 0x35),
    "wrist_roll": (0x26, 0x36),
    "gripper": (0x27, 0x37),
}
FOLLOWER_JOINT_DIRECTIONS = {
    "shoulder_pan": -1.0,
    "shoulder_lift": -1.0,
    "elbow_flex": 1.0,
    "wrist_flex": 1.0,
    "wrist_yaw": 1.0,
    "wrist_roll": -1.0,
    "gripper": -1.0,
}

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]

_passed = 0
_failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


def main():
    print("=" * 60)
    print("mock master-slave following test (no hardware)")
    print("=" * 60)

    bus = FakeBus()
    leader = SeeedB601DMLeader(SeeedB601DMLeaderConfig(id="master"), bus=bus)
    follower = SeeedB601DMFollower(
        SeeedB601DMFollowerConfig(
            id="follower",
            port="",
            can_adapter="dm_device",
            motor_can_ids=FOLLOWER_IDS,
            joint_directions=FOLLOWER_JOINT_DIRECTIONS,
        ),
        bus=bus,
    )

    leader.connect(calibrate=False)
    follower.connect(calibrate=False)

    # --- 1. leader reads 7 joints ---
    print("\n[1] leader read")
    action = leader.get_action()
    expected = {f"{m}.pos" for m in leader.motor_names}
    check("get_action returns all 7 joint positions", set(action.keys()) == expected)

    # --- 2. gravity compensation ---
    print("\n[2] gravity compensation")
    leader.start_gravity_compensation()
    check("6 arm + gripper enabled", all(leader.motors[n]._enabled for n in ARM + ["gripper"]))
    check("gripper switched to MIT mode", leader.motors["gripper"]._mode == Mode.MIT)

    # neutral pose -> elbow should command ~ -7 N·m (DevArm masses)
    leader.gravity_tick()
    elbow_tau = leader.motors["elbow_flex"].last_mit[4]
    check(f"elbow gravity tau ~ -7 N·m (got {elbow_tau:.2f})", abs(elbow_tau - (-7.18)) < 0.5)

    # vertical joints -> ~0 torque
    check("shoulder_pan gravity tau ~ 0", abs(leader.motors["shoulder_pan"].last_mit[4]) < 0.05)

    # gripper free-drive -> tau == 0
    check("gripper free-drive tau == 0", leader.motors["gripper"].last_mit[4] == 0.0)

    # --- 3. follow: leader pos -> follower command (direction mapping) ---
    print("\n[3] follow mapping")
    for i, name in enumerate(leader.motor_names):
        leader.motors[name].pos = 0.1 * (i + 1)  # radians
    action = leader.get_action()
    follower.send_action(action)

    sp_deg = action["shoulder_pan.pos"]
    sp_expected_rad = math.radians(sp_deg * FOLLOWER_JOINT_DIRECTIONS["shoulder_pan"])
    sp_actual_rad = follower.motors["shoulder_pan"].last_pos_vel[0]
    check(
        "shoulder_pan: follower target = leader deg * direction",
        abs(sp_actual_rad - sp_expected_rad) < 1e-9,
    )

    gr_deg = action["gripper.pos"]
    gr_expected = gr_deg * FOLLOWER_JOINT_DIRECTIONS["gripper"]
    gr_actual_rad = follower.motors["gripper"].last_force_pos[0]
    check(
        "gripper: follower target = leader deg * direction (FORCE_POS)",
        abs(gr_actual_rad - math.radians(gr_expected)) < 1e-9,
    )

    # --- 4. stop gravity compensation ---
    print("\n[4] stop gravity compensation")
    leader.stop_gravity_compensation()
    check("arm + gripper disabled after stop", not any(leader.motors[n]._enabled for n in ARM + ["gripper"]))

    # --- 4b. official lerobot-record lifecycle adapter ---
    print("\n[4b] lerobot-record lifecycle adapter")
    record_bus = FakeBus()
    record_leader = SeeedB601DMLeader(
        SeeedB601DMLeaderConfig(
            id="record_master",
            calibrate_on_connect=False,
            auto_gravity_compensation=True,
            max_startup_gripper_offset_deg=90.0,
            action_filter_alpha={"shoulder_pan.pos": 0.5},
            action_filter_deadband={"shoulder_pan.pos": 0.0},
        ),
        bus=record_bus,
    )
    record_leader.connect(calibrate=False)
    check("record leader auto-starts gravity compensation", record_leader._gc_active)
    record_leader.motors["shoulder_pan"].pos = math.radians(10.0)
    filtered_action = record_leader.get_action()
    check("recorded action uses configured low-pass filter", abs(filtered_action["shoulder_pan.pos"] - 5.0) < 1e-6)
    record_motor_refs = list(record_leader.motors.values())
    record_leader.disconnect()
    check("record leader disconnect disables all motors", not any(m._enabled for m in record_motor_refs))

    record_follower_bus = FakeBus()
    record_follower = SeeedB601DMFollower(
        SeeedB601DMFollowerConfig(
            id="record_follower",
            port="",
            can_adapter="dm_device",
            motor_can_ids=FOLLOWER_IDS,
            calibrate_on_connect=False,
            enable_motors_on_connect=True,
            safe_zero_on_disconnect=False,
        ),
        bus=record_follower_bus,
    )
    record_follower.connect(calibrate=False)
    check("record follower auto-enables all motors", all(m._enabled for m in record_follower.motors.values()))
    record_follower.disconnect()

    print("\n" + "=" * 60)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 60)
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
