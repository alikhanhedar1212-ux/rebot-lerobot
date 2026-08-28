"""Analytic gravity compensation for the Seeed reBot B601-DM arm.

Computes the generalized gravity torque vector g(q) for the 6 arm joints from the
arm's URDF mass / centre-of-mass data, using only NumPy (no Pinocchio). It is the
``tau = J^T * m * g`` formulation and matches Pinocchio's
``computeGeneralizedGravity`` (which is RNEA with v=0, a=0).

The 6 arm joints, in order, map to the motors as::

    q[0] shoulder_pan
    q[1] shoulder_lift
    q[2] elbow_flex
    q[3] wrist_flex
    q[4] wrist_yaw
    q[5] wrist_roll

Angles are in **radians**, torques in **N*m**.
"""

from __future__ import annotations

import numpy as np

# Gravity in the world/base frame (m/s^2). The arm's base sits upright, so the
# gravity acceleration points along -Z.
#
# NOTE: the returned torque is the COMPENSATION torque (what the motor must apply
# to hold the arm against gravity), i.e. the NEGATIVE of the raw gravity-force
# torque. This matches Pinocchio's `computeGeneralizedGravity` (with
# model.gravity.linear = (0,0,-9.81)), which is what the ROS2 gravity-compensation
# path feeds to the motors. The `+9.81` below produces that sign.
GRAVITY = np.array([0.0, 0.0, 9.81], dtype=np.float64)

# Chain definition. Each entry is one link following its parent link, described by
# the joint that connects them (URDF <origin> xyz/rpy + <axis>), the child link's
# mass, and the child link's centre of mass in the child-link frame.
#
# Data is from reBot-DevArm_fixend.urdf — the URDF the ROS2 gravity-compensation
# path actually loads. Its link masses INCLUDE the Damiao motor self-weight, so
# they are ~2-3x the reBot_B601_DM_with_gripper.urdf masses; using them here
# removes the need for the old GRAVITY_GAIN / tau_scale fudge factors.
#
# Order: link1..link6 (6 revolute arm joints), then a single lumped gripper_link
# (fixed joint). DevArm has no separate prismatic finger links.
_CHAIN = [
    # joint1 (shoulder_pan)  base_link -> link1
    {
        "origin": np.array([-8.416e-5, 0.0, 0.08465]),
        "rpy": np.array([0.0, 0.0, 0.0]),
        "axis": np.array([0.0, 0.0, 1.0]),
        "joint_type": "revolute",
        "mass": 0.1613,
        "com": np.array([0.000113614552951627, -0.000616319527051323, 0.0236476372671394]),
    },
    # joint2 (shoulder_lift)  link1 -> link2
    {
        "origin": np.array([0.020084, 0.031625, 0.05555]),
        "rpy": np.array([-1.5708, 0.0, 0.0]),
        "axis": np.array([0.0, 0.0, -1.0]),
        "joint_type": "revolute",
        "mass": 1.3266,
        "com": np.array([-0.13225622308888, -0.0030617036386309, -0.0308306967030205]),
    },
    # joint3 (elbow_flex)  link2 -> link3
    {
        "origin": np.array([-0.264, 0.0, 0.0]),
        "rpy": np.array([0.0, 0.0, 0.0]),
        "axis": np.array([0.0, 0.0, 1.0]),
        "joint_type": "revolute",
        "mass": 0.8353,
        "com": np.array([0.121040035791843, -0.0536211076627949, -0.0310137854608077]),
    },
    # joint4 (wrist_flex)  link3 -> link4
    {
        "origin": np.array([0.2426, -0.054, -0.001625]),
        "rpy": np.array([0.0, 0.0, 0.0]),
        "axis": np.array([0.0, 0.0, 1.0]),
        "joint_type": "revolute",
        "mass": 0.52,
        "com": np.array([0.0608200956293136, -0.0511711906613122, -0.030299458623927]),
    },
    # joint5 (wrist_yaw)  link4 -> link5
    {
        "origin": np.array([0.078308, -0.0375, -0.03]),
        "rpy": np.array([-1.5708, 0.0, 0.0]),
        "axis": np.array([0.0, 0.0, 1.0]),
        "joint_type": "revolute",
        "mass": 0.383,
        "com": np.array([-0.00502802058982517, 1.73866206692364e-6, 0.0386233236326755]),
    },
    # joint6 (wrist_roll)  link5 -> link6
    {
        "origin": np.array([0.028008, 0.0, 0.04]),
        "rpy": np.array([0.0, 1.5708, 0.0]),
        "axis": np.array([0.0, 0.0, 1.0]),
        "joint_type": "revolute",
        "mass": 0.3663,
        "com": np.array([3.76418727127126e-6, -0.000100908819946677, 0.0253308606425965]),
    },
    # gripper_joint (fixed)  link6 -> gripper_link
    {
        "origin": np.array([0.0, 0.0, 0.15539]),
        "rpy": np.array([0.0, -1.5708, 3.1415]),
        "axis": np.array([0.0, 0.0, 0.0]),
        "joint_type": "fixed",
        "mass": 0.5,
        "com": np.array([-0.0737654295815033, -9.5080995865868e-6, 7.04327840286845e-6]),
    },
]

N_ARM_JOINTS = 6


def _rpy_to_rotation(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw -> rotation matrix (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))."""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _axis_rotation(axis: np.ndarray, q: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit axis by angle q."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    kx, ky, kz = axis
    k = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + np.sin(q) * k + (1.0 - np.cos(q)) * (k @ k)


def compute_gravity_torque(q: np.ndarray) -> np.ndarray:
    """Compute the gravity torque vector g(q) for the 6 arm joints.

    Args:
        q: array-like of 6 arm joint angles in radians, in the order
           [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll].

    Returns:
        np.ndarray of shape (6,) with the gravity torque in N*m.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape[0] < N_ARM_JOINTS:
        raise ValueError(f"expected {N_ARM_JOINTS} joint angles, got {q.shape[0]}")

    # Forward kinematics: walk the chain, tracking each joint's world origin/axis
    # and each link's world centre of mass.
    T = np.eye(4)  # current link world transform (starts at base_link = identity)

    joint_origins = []  # world origin of each of the 6 arm joints
    joint_axes = []     # world axis of each of the 6 arm joints
    link_coms = []      # world centre of mass of every link (9 total)
    link_masses = []

    for idx, link in enumerate(_CHAIN):
        origin = link["origin"]
        rpy = link["rpy"]
        axis = link["axis"]
        jtype = link["joint_type"]

        R_origin = _rpy_to_rotation(rpy)
        T_joint = T @ np.vstack(
            [np.hstack([R_origin, origin.reshape(3, 1)]), [0.0, 0.0, 0.0, 1.0]]
        )

        # World origin and axis of this joint (before applying the joint rotation).
        o_j = T_joint[:3, 3]
        a_j = T_joint[:3, :3] @ axis

        if jtype == "revolute" and idx < N_ARM_JOINTS:
            joint_origins.append(o_j)
            joint_axes.append(a_j)
            T = T_joint @ np.vstack(
                [np.hstack([_axis_rotation(axis, q[idx]), np.zeros((3, 1))]), [0.0, 0.0, 0.0, 1.0]]
            )
        else:
            # fixed or prismatic-at-zero: no rotation / no displacement.
            T = T_joint

        # Child link's centre of mass in the world frame.
        com_world = (T @ np.append(link["com"], 1.0))[:3]
        link_coms.append(com_world)
        link_masses.append(link["mass"])

    # tau_j = sum over links downstream of joint j of  m_i * g . (a_j x (p_i - o_j))
    tau = np.zeros(N_ARM_JOINTS)
    for j in range(N_ARM_JOINTS):
        a_j = joint_axes[j]
        o_j = joint_origins[j]
        # Links downstream of arm joint j are all links from index j onward
        # (link indices match joint indices for the 6 arm links; the gripper links
        # 6..8 are always downstream of every arm joint).
        for i in range(j, len(link_coms)):
            r = link_coms[i] - o_j
            lever = np.cross(a_j, r)
            tau[j] += link_masses[i] * float(GRAVITY @ lever)

    return tau
