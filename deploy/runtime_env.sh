#!/usr/bin/env bash
# Shared relocatable runtime environment. Safe to source interactively.

DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$DEPLOY_DIR/.." && pwd)"
ENV_DIR="${REBOT_ENV_DIR:-$PROJECT_DIR/.conda/envs/lerobot-follow}"
MACHINE_CONFIG="${REBOT_MACHINE_CONFIG:-$PROJECT_DIR/config/machine.env}"
CAMERA_CONFIG="${REBOT_CAMERA_CONFIG:-$PROJECT_DIR/config/cameras.json}"

if [[ -r "$MACHINE_CONFIG" ]]; then
  # shellcheck disable=SC1090
  source "$MACHINE_CONFIG"
fi

export PROJECT_DIR ENV_DIR MACHINE_CONFIG CAMERA_CONFIG
export REBOT_LEADER_INDEX="${REBOT_LEADER_INDEX:-1}"
export REBOT_FOLLOWER_INDEX="${REBOT_FOLLOWER_INDEX:-0}"
export REBOT_DM_DEVICE_PRIME_INDEX="${REBOT_DM_DEVICE_PRIME_INDEX:-$REBOT_LEADER_INDEX}"
export CONDA_PREFIX="$ENV_DIR"
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/lerobot/src:$PROJECT_DIR/lerobot-robot-seeed-b601${PYTHONPATH:+:$PYTHONPATH}"
export MOTOR_DM_DEVICE_LIB="$PROJECT_DIR/motorbridge-sdk-work/third_party/dm_device/v1.1.0/linux/x86_64/libdm_device.so"

runtime_libs="$ENV_DIR/lib"
if [[ "${REBOT_USE_MVS_LIBS:-0}" == 1 ]]; then
  runtime_libs="$runtime_libs:/opt/MVS/lib/64:/opt/MVS/lib/32"
fi
export LD_LIBRARY_PATH="$runtime_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Missing project environment: $ENV_DIR" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -r "$MOTOR_DM_DEVICE_LIB" ]]; then
  echo "Missing DM Device SDK: $MOTOR_DM_DEVICE_LIB" >&2
  return 1 2>/dev/null || exit 1
fi
