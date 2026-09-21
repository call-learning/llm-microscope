#!/usr/bin/env bash
# Set up a uv-managed virtual environment for llm-microscope with a CUDA PyTorch.
#
# Usage:
#   ./scripts/setup.sh                        # base + captum + umap
#   ./scripts/setup.sh --all                  # install every optional extra
#   ./scripts/setup.sh --torch-index URL      # e.g. https://download.pytorch.org/whl/cu124
#   UV_TORCH_INDEX_URL=... ./scripts/setup.sh
#
# The script keeps any working PyTorch that already exists on the machine
# (system interpreter or a previous .venv) and only installs one from the
# PyTorch wheel index when nothing suitable is found.
set -euo pipefail

cd "$(dirname "$0")/.."

EXTRAS="captum,umap"
TORCH_INDEX_URL="${UV_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      EXTRAS="all"
      shift
      ;;
    --torch-index)
      TORCH_INDEX_URL="${2:?--torch-index requires a URL}"
      shift 2
      ;;
    --torch-index=*)
      TORCH_INDEX_URL="${1#*=}"
      shift
      ;;
    --help|-h)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

# 1. uv itself
if ! command -v uv >/dev/null 2>&1; then
  echo "==> uv not found; installing via official installer"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "==> uv $(uv --version | awk '{print $2}')"

detect_torch() {
  local py="$1"
  "$py" - <<'EOF' 2>/dev/null || true
import sys, importlib.util
spec = importlib.util.find_spec("torch")
if spec is None:
    sys.exit(1)
import torch
print(f"{torch.__version__}|{torch.cuda.is_available()}")
EOF
}

# Pick a python interpreter that satisfies project.requires-python (>=3.11,<3.13).
# Prefers uv-managed CPythons over the system python.
pick_python() {
  local want
  want="$(uv python find 3.12 2>/dev/null || uv python find 3.11 2>/dev/null || true)"
  if [[ -n "${want}" && -x "${want}" ]]; then
    echo "${want}"
    return
  fi
  if python3 -c 'import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] < (3,13) else 1)' 2>/dev/null; then
    echo "python3"
    return
  fi
  echo ""
}

# 2. Decide where the PyTorch comes from.
#    a) Existing .venv already has a CUDA torch -> keep the venv.
#    b) System python has a CUDA torch -> recreate the venv with
#       system site packages and pin that exact torch wheel.
#    c) Otherwise install a CUDA torch from the wheel index into a fresh venv.
PYTHON="$(pick_python)"
if [[ -z "${PYTHON}" ]]; then
  echo "error: no Python 3.11+ interpreter found. Install one (e.g. 'uv python install 3.12')." >&2
  exit 1
fi

torch_source=""
existing_torch=""
if [[ -x .venv/bin/python ]]; then
  existing_torch="$(detect_torch .venv/bin/python || true)"
fi
if [[ -z "${existing_torch}" && -x .venv/bin/python ]]; then
  echo "==> Removing existing .venv without a usable PyTorch"
  rm -rf .venv
fi

if [[ -n "${existing_torch}" ]]; then
  torch_source="venv"
  echo "==> Keeping existing PyTorch ${existing_torch%%|*} from .venv"
else
  sys_torch="$(detect_torch "${PYTHON}" || true)"
  if [[ -n "${sys_torch}" && "${sys_torch##*|}" == "True" ]]; then
    torch_source="system"
    echo "==> Found system PyTorch ${sys_torch%%|*} (${PYTHON}); building venv around it"
  else
    torch_source="fresh"
    echo "==> No working PyTorch found; installing CUDA build from ${TORCH_INDEX_URL}"
  fi
fi

# 3. Create the virtual environment.
if [[ "${torch_source}" == "venv" ]]; then
  :
elif [[ "${torch_source}" == "system" ]]; then
  uv venv --system-site-packages -p "${PYTHON}"
  uv pip install --python ".venv/bin/python" \
    "torch==${sys_torch%%|*}" \
    --index-url "${TORCH_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match
else
  uv venv -p 3.12
  uv pip install --python ".venv/bin/python" torch --index-url "${TORCH_INDEX_URL}"
fi

# 4. Project dependencies + selected extras.
#    torch is not a project dependency, so uv sync never touches the
#    environment's existing CUDA build.
sync_args=()
if [[ "${EXTRAS}" == "all" ]]; then
  sync_args+=(--all-extras)
else
  for extra in ${EXTRAS//,/ }; do
    sync_args+=(--extra "${extra}")
  done
fi
uv sync "${sync_args[@]}"

# 5. Verify.
echo
.venv/bin/python scripts/check_environment.py
echo
echo "==> Setup complete."
echo "    Run the app with:  uv run streamlit run app.py"
echo "    (or: source .venv/bin/activate && streamlit run app.py)"
