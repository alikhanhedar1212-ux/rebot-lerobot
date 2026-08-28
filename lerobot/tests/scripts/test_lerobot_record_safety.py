from lerobot.scripts.lerobot_record import SingleGraspSafetyWatchdog, _run_policy_safety_return


def observation(arm: float = 0.0, gripper: float = 0.0) -> dict[str, float]:
    return {
        "shoulder_pan.pos": arm,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 0.0,
        "wrist_flex.pos": 0.0,
        "wrist_yaw.pos": 0.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": gripper,
    }


def watchdog() -> SingleGraspSafetyWatchdog:
    return SingleGraspSafetyWatchdog(
        stationary_timeout_s=5.0,
        motion_delta_deg=0.5,
        start_delta_deg=3.0,
        gripper_closed_deg=80.0,
        gripper_open_deg=20.0,
    )


def test_start_pose_is_excluded_from_stationary_timeout() -> None:
    guard = watchdog()
    assert guard.update(observation(), 0.0) is None
    assert guard.update(observation(arm=2.9), 100.0) is None


def test_mid_rollout_stationary_timeout_after_five_seconds() -> None:
    guard = watchdog()
    assert guard.update(observation(), 0.0) is None
    assert guard.update(observation(arm=4.0), 1.0) is None
    assert guard.update(observation(arm=4.2), 5.9) is None
    assert guard.update(observation(arm=4.2), 6.0) == "stationary_timeout"


def test_cumulative_motion_resets_stationary_timer() -> None:
    guard = watchdog()
    guard.update(observation(), 0.0)
    guard.update(observation(arm=4.0), 1.0)
    assert guard.update(observation(arm=4.6), 5.5) is None
    assert guard.update(observation(arm=4.6), 10.4) is None
    assert guard.update(observation(arm=4.6), 10.5) == "stationary_timeout"


def test_first_close_then_reopen_enters_end_phase_without_return() -> None:
    guard = watchdog()
    guard.update(observation(), 0.0)
    guard.update(observation(arm=4.0), 1.0)
    assert guard.update(observation(arm=5.0, gripper=-90.0), 2.0) is None
    assert guard.gripper_closed_once
    assert guard.update(observation(arm=6.0, gripper=-30.0), 3.0) is None
    assert guard.update(observation(arm=6.0, gripper=-19.0), 4.0) is None
    assert guard.ended
    # End state is excluded even after a long stationary interval.
    assert guard.update(observation(arm=6.0, gripper=0.0), 100.0) is None


def test_second_gripper_close_is_blocked_before_send() -> None:
    guard = watchdog()
    guard.update(observation(), 0.0)
    guard.update(observation(arm=4.0), 1.0)
    guard.update(observation(arm=5.0, gripper=-90.0), 2.0)
    guard.update(observation(arm=6.0, gripper=0.0), 3.0)
    assert guard.check_action({"gripper.pos": -20.0}) is None
    assert guard.check_action({"gripper.pos": -90.0}) == "second_grasp_blocked"


def test_safety_return_calls_safe_zero() -> None:
    class FakeRobot:
        returned = False

        def safe_zero(
            self,
            step_interval_s: float,
            exit_on_complete: bool,
            velocity_scale: float,
        ) -> None:
            assert step_interval_s == 0.012
            assert velocity_scale == 1.30
            assert not exit_on_complete
            self.returned = True

    robot = FakeRobot()
    _run_policy_safety_return(robot, "stationary_timeout", 0.012, 1.30)  # type: ignore[arg-type]
    assert robot.returned
