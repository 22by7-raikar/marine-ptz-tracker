#!/bin/sh
# Compile the Uno sketch only; installation and uploads are intentionally absent.
set -eu

EXPECTED_CLI_VERSION="1.5.1"
EXPECTED_CORE="arduino:avr@1.8.8"
EXPECTED_SERVO="Servo@1.3.0"
FQBN="arduino:avr:uno"

if [ "$#" -gt 1 ]; then
  echo "usage: $0 [build-directory]" >&2
  exit 2
fi

if [ -n "${ARDUINO_CLI:-}" ]; then
  cli="$ARDUINO_CLI"
else
  cli="arduino-cli"
fi
if ! command -v "$cli" >/dev/null 2>&1; then
  echo "Arduino CLI is required; install Arduino CLI ${EXPECTED_CLI_VERSION} first." >&2
  exit 1
fi

run_cli() {
  if [ -n "${ARDUINO_CLI_CONFIG_DIR:-}" ]; then
    "$cli" --config-dir "$ARDUINO_CLI_CONFIG_DIR" "$@"
  else
    "$cli" "$@"
  fi
}

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
sketch_dir="${repository_root}/firmware/ptz_controller"
if [ ! -d "$sketch_dir" ]; then
  echo "firmware sketch directory is missing: $sketch_dir" >&2
  exit 1
fi

created_temporary_build=0
if [ "$#" -eq 1 ]; then
  build_dir=$1
else
  build_dir=$(mktemp -d "${TMPDIR:-/tmp}/marine-ptz-arduino-build.XXXXXX")
  created_temporary_build=1
fi

cleanup() {
  if [ "$created_temporary_build" -eq 1 ]; then
    rm -rf -- "$build_dir"
  fi
}
trap cleanup EXIT HUP INT TERM

if version_output=$(run_cli version 2>&1); then
  :
else
  version_status=$?
  printf '%s\n' "$version_output" >&2
  echo "could not query Arduino CLI version" >&2
  exit "$version_status"
fi
printf '%s\n' "$version_output"
version_fields=$(printf '%s\n' "$version_output" | awk '
  {
    for (field_index = 1; field_index <= NF; field_index++) {
      if ($field_index == "Version:") {
        if (field_index == NF) {
          print "<empty>"
        } else {
          print $(field_index + 1)
        }
      }
    }
  }
')
version_field_count=$(printf '%s\n' "$version_output" | awk '
  {
    for (field_index = 1; field_index <= NF; field_index++) {
      if ($field_index == "Version:") {
        count++
      }
    }
  }
  END { print count + 0 }
')

actual_version=$version_fields
if [ "$version_field_count" -ne 1 ]; then
  if [ "$version_field_count" -eq 0 ]; then
    actual_version="<missing>"
  else
    actual_version="<multiple: ${version_fields}>"
  fi
elif ! printf '%s\n' "$actual_version" | awk '
  NR == 1 && $0 ~ /^[0-9]+\.[0-9]+\.[0-9]+$/ { valid = 1 }
  END { exit valid ? 0 : 1 }
'; then
  actual_version="<malformed: ${actual_version:-<empty>}>"
fi

if [ "$actual_version" != "$EXPECTED_CLI_VERSION" ]; then
  echo "Arduino CLI version mismatch: expected ${EXPECTED_CLI_VERSION}, got ${actual_version}." >&2
  exit 1
fi

if ! run_cli core list | grep -Eq "arduino:avr[[:space:]]+1\\.8\\.8([[:space:]]|$)"; then
  echo "Arduino core ${EXPECTED_CORE} is required; install it before compiling." >&2
  exit 1
fi
if ! run_cli lib list | grep -Eq "Servo[[:space:]]+1\\.3\\.0([[:space:]]|$)"; then
  echo "Arduino library ${EXPECTED_SERVO} is required; install it before compiling." >&2
  exit 1
fi

if [ -n "${ARDUINO_CLI_CONFIG_DIR:-}" ]; then
  printf '+ %s --config-dir %s compile --fqbn %s --warnings all --build-path %s %s\n' \
    "$cli" "$ARDUINO_CLI_CONFIG_DIR" "$FQBN" "$build_dir" "$sketch_dir"
else
  printf '+ %s compile --fqbn %s --warnings all --build-path %s %s\n' \
    "$cli" "$FQBN" "$build_dir" "$sketch_dir"
fi
run_cli compile \
  --fqbn "$FQBN" \
  --warnings all \
  --build-path "$build_dir" \
  "$sketch_dir"
