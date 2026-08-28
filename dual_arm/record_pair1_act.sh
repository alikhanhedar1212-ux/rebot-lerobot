#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../deploy/runtime_env.sh
source "$SCRIPT_DIR/../deploy/runtime_env.sh"
CONFIG_PATH="$PROJECT_DIR/dual_arm/act_pair1_record.json"
DATASET_ROOT="$PROJECT_DIR/data/act_green_to_paper"
D405_COLOR="${REBOT_D405_COLOR:?Set REBOT_D405_COLOR in config/machine.env}"
D435I_COLOR="${REBOT_D435I_COLOR:?Set REBOT_D435I_COLOR in config/machine.env}"
export LEROBOT_VIDEO_CRF=20

if pgrep -x realsense-viewer >/dev/null; then
  echo "Close RealSense Viewer before recording; it currently owns the cameras." >&2
  exit 2
fi

RESUME_REQUESTED=false
for arg in "$@"; do
  if [[ "$arg" == "--resume=true" ]]; then
    RESUME_REQUESTED=true
  elif [[ "$arg" == --dataset.root=* ]]; then
    DATASET_ROOT="${arg#--dataset.root=}"
  fi
done

if [[ -e "$DATASET_ROOT" && "$RESUME_REQUESTED" != true ]]; then
  echo "Dataset path already exists: $DATASET_ROOT" >&2
  echo "Choose a new ACT_DATASET_ROOT, or add --resume=true when intentionally resuming." >&2
  exit 4
fi

# Keep exposure and sharpness reproducible across USB reconnects. D405 does
# not expose manual color exposure through its V4L2 node, so use a small
# brightness correction there. D435i supports a fixed exposure directly.
v4l2-ctl -d "$D405_COLOR" --set-ctrl=brightness=-5,sharpness=65
v4l2-ctl -d "$D435I_COLOR" --set-ctrl=auto_exposure=1,exposure_time_absolute=100,sharpness=65

cd "$PROJECT_DIR"
exec "$ENV_DIR/bin/lerobot-record" \
  --config_path="$CONFIG_PATH" \
  --robot.dm_device_index="$REBOT_FOLLOWER_INDEX" \
  --teleop.dm_device_index="$REBOT_LEADER_INDEX" \
  --robot.cameras="$(tr -d '\n' < "$CAMERA_CONFIG")" \
  --dataset.root="$DATASET_ROOT" \
  "$@"
