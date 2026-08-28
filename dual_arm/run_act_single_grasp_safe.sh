#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../deploy/runtime_env.sh
source "$SCRIPT_DIR/../deploy/runtime_env.sh"
POLICY_DIR="$PROJECT_DIR/outputs/act_green_to_paper_bs2_50k/checkpoints/050000/pretrained_model"
CONFIG_PATH="$PROJECT_DIR/dual_arm/act_pair1_record.json"
CAMERAS_JSON="$(tr -d '\n' < "$CAMERA_CONFIG")"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <unique-run-id>  (example: 003_safe)" >&2
  exit 2
fi

RUN_ID=$1
if [[ ! $RUN_ID =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Run ID may contain only letters, numbers, '_' and '-'." >&2
  exit 2
fi

DATASET_NAME="eval_act_green_to_paper_${RUN_ID}"
DATASET_ROOT="$PROJECT_DIR/data/$DATASET_NAME"
if [[ -e $DATASET_ROOT ]]; then
  echo "Dataset path already exists: $DATASET_ROOT" >&2
  echo "Choose a new run ID; existing evaluation data will not be overwritten." >&2
  exit 3
fi

cd "$PROJECT_DIR"
exec "$ENV_DIR/bin/lerobot-record" \
  --config_path="$CONFIG_PATH" \
  --policy.path="$POLICY_DIR" \
  --teleop=null \
  --robot.dm_device_index="$REBOT_FOLLOWER_INDEX" \
  --robot.pos_vel_velocity='[46.4,46.4,57.6,38.4,31.2,46.4,80.0]' \
  --robot.cameras="$CAMERAS_JSON" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.repo_id="local/$DATASET_NAME" \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=5 \
  --manual_episode_control=false \
  --display_data=true \
  --safety_single_grasp=true \
  --safety_stationary_timeout_s=5.0 \
  --safety_motion_delta_deg=0.5 \
  --safety_start_delta_deg=3.0 \
  --safety_gripper_closed_deg=80.0 \
  --safety_gripper_open_deg=20.0 \
  --safety_zero_step_interval_s=0.012 \
  --safety_zero_velocity_scale=1.625
