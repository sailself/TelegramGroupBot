#!/usr/bin/env bash
set -euo pipefail

BINARY_NAME="${BINARY_NAME:-telegram_group_helper_bot}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_DIR="${APP_DIR:-/opt/telegram_chat_bot}"
SERVICE_NAME="${SERVICE_NAME:-telegram_group_helper_bot}"
SERVICE_USER="${SERVICE_USER:-${SUDO_USER:-$USER}}"
SYSTEMD_DIR="${SYSTEMD_DIR:-/etc/systemd/system}"
INSTALL_SYSTEMD="${INSTALL_SYSTEMD:-1}"
START_SERVICE="${START_SERVICE:-1}"

SOURCE_BINARY="${BUNDLE_ROOT}/target/release/${BINARY_NAME}"
SOURCE_ENV_EXAMPLE="${BUNDLE_ROOT}/.env.example"
SOURCE_README="${BUNDLE_ROOT}/README.md"
SOURCE_SERVICE_TEMPLATE="${BUNDLE_ROOT}/deploy/telegram_group_helper_bot.service"
SOURCE_INSTALL_SCRIPT="${BUNDLE_ROOT}/deploy/install_release_bundle.sh"

if [[ ! -f "${SOURCE_BINARY}" ]]; then
  echo "Release binary not found in bundle: ${SOURCE_BINARY}" >&2
  exit 1
fi

if [[ "${INSTALL_SYSTEMD}" == "1" && "${EUID}" -ne 0 ]]; then
  echo "INSTALL_SYSTEMD=1 requires root privileges. Re-run with sudo or set INSTALL_SYSTEMD=0." >&2
  exit 1
fi

mkdir -p \
  "${APP_DIR}" \
  "${APP_DIR}/deploy" \
  "${APP_DIR}/target/release" \
  "${APP_DIR}/data" \
  "${APP_DIR}/logs" \
  "${APP_DIR}/run"

install -m 755 "${SOURCE_BINARY}" "${APP_DIR}/target/release/${BINARY_NAME}"
install -m 644 "${SOURCE_ENV_EXAMPLE}" "${APP_DIR}/.env.example"
install -m 644 "${SOURCE_README}" "${APP_DIR}/README.md"
install -m 644 "${SOURCE_SERVICE_TEMPLATE}" "${APP_DIR}/deploy/${SERVICE_NAME}.service.template"
install -m 755 "${SOURCE_INSTALL_SCRIPT}" "${APP_DIR}/deploy/install_release_bundle.sh"

rendered_service_path="${APP_DIR}/deploy/${SERVICE_NAME}.service"
sed \
  -e "s|^User=.*$|User=${SERVICE_USER}|" \
  -e "s|^WorkingDirectory=.*$|WorkingDirectory=${APP_DIR}|" \
  -e "s|^ExecStart=.*$|ExecStart=${APP_DIR}/target/release/${BINARY_NAME}|" \
  "${SOURCE_SERVICE_TEMPLATE}" > "${rendered_service_path}"

echo "Installed release bundle into ${APP_DIR}"
echo "Rendered service template to ${rendered_service_path}"

if [[ ! -f "${APP_DIR}/.env" ]]; then
  echo "No ${APP_DIR}/.env found."
  echo "Create it from ${APP_DIR}/.env.example before starting the service."
fi

if [[ "${INSTALL_SYSTEMD}" != "1" ]]; then
  echo "Skipping systemd installation because INSTALL_SYSTEMD=${INSTALL_SYSTEMD}"
  exit 0
fi

install -m 644 "${rendered_service_path}" "${SYSTEMD_DIR}/${SERVICE_NAME}.service"
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"

if [[ "${START_SERVICE}" != "1" ]]; then
  echo "Installed and enabled ${SERVICE_NAME}. Start it manually when ready."
  exit 0
fi

if [[ ! -f "${APP_DIR}/.env" ]]; then
  echo "Skipped starting ${SERVICE_NAME} because ${APP_DIR}/.env is missing."
  exit 0
fi

if systemctl is-active --quiet "${SERVICE_NAME}"; then
  systemctl restart "${SERVICE_NAME}"
  echo "Restarted ${SERVICE_NAME}"
else
  systemctl start "${SERVICE_NAME}"
  echo "Started ${SERVICE_NAME}"
fi
