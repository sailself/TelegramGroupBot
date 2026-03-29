#!/usr/bin/env bash
set -euo pipefail

BINARY_NAME="${BINARY_NAME:-telegram_group_helper_bot}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY_PATH="${REPO_ROOT}/target/release/${BINARY_NAME}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/run}"
PID_FILE="${PID_FILE:-${RUN_DIR}/${BINARY_NAME}.pid}"
NOHUP_LOG="${NOHUP_LOG:-${LOG_DIR}/nohup.bot.log}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
STOP_TIMEOUT_SECONDS="${STOP_TIMEOUT_SECONDS:-30}"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"
cd "${REPO_ROOT}"

if [[ -z "${RUSTFLAGS:-}" ]]; then
  export RUSTFLAGS="-C debuginfo=0"
fi
export CARGO_BUILD_JOBS="${BUILD_JOBS}"

stop_existing_process() {
  if [[ ! -f "${PID_FILE}" ]]; then
    return
  fi

  local pid
  pid="$(cat "${PID_FILE}")"

  if [[ -z "${pid}" ]]; then
    rm -f "${PID_FILE}"
    return
  fi

  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "Removing stale PID file ${PID_FILE}"
    rm -f "${PID_FILE}"
    return
  fi

  echo "Stopping ${BINARY_NAME} (pid ${pid})"
  kill "${pid}"

  local elapsed=0
  while kill -0 "${pid}" 2>/dev/null; do
    if (( elapsed >= STOP_TIMEOUT_SECONDS )); then
      echo "Process did not exit in ${STOP_TIMEOUT_SECONDS}s, sending SIGKILL"
      kill -9 "${pid}"
      break
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  rm -f "${PID_FILE}"
}

if [[ "${SKIP_GIT_PULL:-0}" != "1" ]]; then
  if [[ -d .git ]]; then
    current_branch="$(git rev-parse --abbrev-ref HEAD)"
    echo "Pulling ${GIT_REMOTE}/${current_branch}"
    git pull --ff-only "${GIT_REMOTE}" "${current_branch}"
  else
    echo "No .git directory found, skipping git pull"
  fi
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo "Building release binary with CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}"
  cargo build --release
fi

if [[ ! -x "${BINARY_PATH}" ]]; then
  echo "Release binary not found: ${BINARY_PATH}" >&2
  exit 1
fi

if [[ "${SKIP_RESTART:-0}" == "1" ]]; then
  echo "Skipping restart because SKIP_RESTART=1"
  exit 0
fi

stop_existing_process

echo "Starting ${BINARY_NAME}"
nohup "${BINARY_PATH}" >> "${NOHUP_LOG}" 2>&1 &
new_pid=$!
echo "${new_pid}" > "${PID_FILE}"

echo "Started ${BINARY_NAME} with pid ${new_pid}"
echo "PID file: ${PID_FILE}"
echo "nohup log: ${NOHUP_LOG}"
