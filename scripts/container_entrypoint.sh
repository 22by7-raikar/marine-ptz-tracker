#!/bin/sh
set -eu

mode=${MARINE_PTZ_PREFLIGHT_MODE:-}
app_root=${MARINE_PTZ_APP_ROOT:-/app}
if [ -n "${YOLO_CONFIG_DIR:-}" ]; then
    case "$YOLO_CONFIG_DIR" in
        /tmp/*) ;;
        *)
            printf 'YOLO_CONFIG_DIR must be an absolute path beneath /tmp\n' >&2
            exit 2
            ;;
    esac
    mkdir -p -m 0700 "$YOLO_CONFIG_DIR" "$YOLO_CONFIG_DIR/Ultralytics"
    chmod 0700 "$YOLO_CONFIG_DIR" "$YOLO_CONFIG_DIR/Ultralytics"
fi
# Files created by the runtime are owner-writable, group-readable, and private
# from all other users. A setgid artifact directory supplies the host artifact
# group without changing the container's primary UID/GID.
umask 027
if [ "$mode" = "test" ]; then
    python "$app_root/tools/runtime_preflight.py" --mode test --json
elif [ "$mode" = "replay" ]; then
    python "$app_root/tools/runtime_preflight.py" --mode replay --json \
        --config "${MARINE_PTZ_CONFIG:?MARINE_PTZ_CONFIG is required}" \
        --model "${MARINE_PTZ_MODEL:?MARINE_PTZ_MODEL is required}" \
        --input-video "${MARINE_PTZ_INPUT_VIDEO:?MARINE_PTZ_INPUT_VIDEO is required}" \
        --output-directory "${MARINE_PTZ_OUTPUT_DIRECTORY:?MARINE_PTZ_OUTPUT_DIRECTORY is required}" \
        --target-class "${MARINE_PTZ_TARGET_CLASS:?MARINE_PTZ_TARGET_CLASS is required}" \
        --confidence "${MARINE_PTZ_CONFIDENCE:?MARINE_PTZ_CONFIDENCE is required}" \
        --device "${MARINE_PTZ_DEVICE:-cpu}" \
        --runtime-mode "${MARINE_PTZ_RUNTIME_MODE:-single}"
elif [ "$mode" = "hardware" ]; then
    arm_argument=
    if [ "${MARINE_PTZ_ARM_HARDWARE:-}" = "true" ]; then
        arm_argument=--arm-hardware
    fi
    python "$app_root/tools/runtime_preflight.py" --mode hardware --json \
        --config "${MARINE_PTZ_CONFIG:?MARINE_PTZ_CONFIG is required}" \
        --model "${MARINE_PTZ_MODEL:?MARINE_PTZ_MODEL is required}" \
        --output-directory "${MARINE_PTZ_OUTPUT_DIRECTORY:?MARINE_PTZ_OUTPUT_DIRECTORY is required}" \
        --target-class "${MARINE_PTZ_TARGET_CLASS:?MARINE_PTZ_TARGET_CLASS is required}" \
        --confidence "${MARINE_PTZ_CONFIDENCE:?MARINE_PTZ_CONFIDENCE is required}" \
        --device "${MARINE_PTZ_DEVICE:-cpu}" \
        --runtime-mode "${MARINE_PTZ_RUNTIME_MODE:-single}" \
        --camera-path "${MARINE_PTZ_CAMERA_DEVICE:?MARINE_PTZ_CAMERA_DEVICE is required}" \
        --serial-path "${MARINE_PTZ_SERIAL_DEVICE:?MARINE_PTZ_SERIAL_DEVICE is required}" \
        $arm_argument
elif [ -n "$mode" ]; then
    printf 'unsupported MARINE_PTZ_PREFLIGHT_MODE: %s\n' "$mode" >&2
    exit 2
fi

# The Python process replaces this shell and receives SIGINT/SIGTERM directly.
exec "$@"
