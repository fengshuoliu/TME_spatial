#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_PY="$SCRIPT_DIR/app.py"
REQUIREMENTS_TXT="$SCRIPT_DIR/requirements.txt"
ENV_NAME="TME_spatial"
VENV_DIR="$SCRIPT_DIR/.tme_spatial_macos_venv"
STREAMLIT_URL="http://localhost:8501"

log() {
  printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$1"
}

run_step() {
  local message="$1"
  shift
  log "$message"
  printf 'Command: '
  printf '%q ' "$@"
  printf '\n'
  "$@"
  log "Completed: $message"
}

ensure_python() {
  if command -v python3.11 >/dev/null 2>&1; then
    printf 'python3.11'
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    printf 'python3'
    return
  fi
  if command -v python >/dev/null 2>&1; then
    printf 'python'
    return
  fi

  if command -v brew >/dev/null 2>&1; then
    run_step "Installing Python 3.11 with Homebrew" brew install python@3.11
    if command -v python3.11 >/dev/null 2>&1; then
      printf 'python3.11'
      return
    fi
  fi

  log 'Python was not found. Install Python 3.11, Miniconda, or Homebrew, then launch the app again.'
  /usr/bin/osascript <<OSA
  display dialog "Python 3.11 was not found on this Mac.\n\nPlease install one of these and then open TME Spatial again:\n- Miniconda / Conda\n- Python 3.11\n- Homebrew" buttons {"OK"} default button "OK" with icon stop
OSA
  exit 1
}

open_streamlit_url() {
  (
    for _ in $(seq 1 30); do
      if /usr/bin/curl -fsS "$STREAMLIT_URL" >/dev/null 2>&1; then
        /usr/bin/open "$STREAMLIT_URL" >/dev/null 2>&1 || true
        exit 0
      fi
      sleep 1
    done
  ) &
}

ensure_conda_env() {
  local conda_cmd="$1"
  log "Conda detected at: $conda_cmd"

  if ! "$conda_cmd" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    run_step "Creating Conda environment $ENV_NAME" "$conda_cmd" create -y -n "$ENV_NAME" python=3.11
  else
    log "Conda environment '$ENV_NAME' already exists."
  fi

  run_step "Upgrading pip tools in Conda environment" "$conda_cmd" run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel
  run_step "Installing app requirements in Conda environment" "$conda_cmd" run -n "$ENV_NAME" python -m pip install -r "$REQUIREMENTS_TXT"

  log "If Streamlit starts successfully, the app will open at $STREAMLIT_URL"
  open_streamlit_url
  exec "$conda_cmd" run --live-stream -n "$ENV_NAME" python -m streamlit run "$APP_PY" --server.headless false
}

ensure_venv_env() {
  local python_cmd="$1"

  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    run_step "Creating Python virtual environment" "$python_cmd" -m venv "$VENV_DIR"
  else
    log "Virtual environment already exists at: $VENV_DIR"
  fi

  run_step "Upgrading pip tools in virtual environment" "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
  run_step "Installing app requirements in virtual environment" "$VENV_DIR/bin/python" -m pip install -r "$REQUIREMENTS_TXT"

  log "If Streamlit starts successfully, the app will open at $STREAMLIT_URL"
  open_streamlit_url
  exec "$VENV_DIR/bin/python" -m streamlit run "$APP_PY" --server.headless false
}

main() {
  clear || true
  log 'TME Spatial macOS launcher'
  log "Project folder: $SCRIPT_DIR"

  if [[ ! -f "$APP_PY" ]]; then
    log "Could not find app.py at $APP_PY"
    exit 1
  fi
  if [[ ! -f "$REQUIREMENTS_TXT" ]]; then
    log "Could not find requirements.txt at $REQUIREMENTS_TXT"
    exit 1
  fi

  if command -v conda >/dev/null 2>&1; then
    ensure_conda_env "$(command -v conda)"
  else
    ensure_venv_env "$(ensure_python)"
  fi
}

main "$@"
