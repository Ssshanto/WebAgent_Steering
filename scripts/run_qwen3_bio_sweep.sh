#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLAN_JSON="${ROOT_DIR}/runtime_state/qwen3_layer_plan.json"
OUT_ROOT="${ROOT_DIR}/results/qwen3_sweep"
VEC_ROOT="${ROOT_DIR}/vectors"
LOG_ROOT="${ROOT_DIR}/logs"
CONDA_ENV="${CONDA_ENV:-steer}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

py() {
  "${PYTHON_BIN}" "$@"
}

if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
PYTHON_BIN="${PYTHON_BIN:-${HOME}/anaconda3/envs/${CONDA_ENV}/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python binary not found for env ${CONDA_ENV}: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${PLAN_JSON}" ]]; then
  mkdir -p "$(dirname "${PLAN_JSON}")"
  py - <<'PY'
import json
from pathlib import Path
from transformers import AutoConfig

models = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
}
plan = {}
for key, model_id in models.items():
    n = int(AutoConfig.from_pretrained(model_id).num_hidden_layers)
    start = (n - 6) // 2
    plan[key] = {
        "model_id": model_id,
        "num_hidden_layers": n,
        "middle6": list(range(start, start + 6)),
        "middle1": [n // 2],
    }

out = Path("runtime_state/qwen3_layer_plan.json")
out.write_text(json.dumps(plan, indent=2) + "\n")
print(out)
PY
fi

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${ROOT_DIR}/runtime_state"

run_cmd() {
  if [[ ${DRY_RUN} -eq 1 ]]; then
    printf '[dry-run] %q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

is_valid_jsonl() {
  local file_path="$1"
  [[ -s "${file_path}" ]] || return 1
  py -c 'import json,sys; p=sys.argv[1];
with open(p,"r",encoding="utf-8") as f:
    [json.loads(line) for line in f if line.strip()]
print("ok")' "${file_path}" >/dev/null 2>&1
}

stage_complete() {
  local summary_path="$1"
  local jsonl_path="$2"
  [[ -f "${summary_path}" ]] && is_valid_jsonl "${jsonl_path}"
}

models=(qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-8b)

for idx in "${!models[@]}"; do
  model="${models[$idx]}"
  next_idx=$((idx + 1))
  next_model=""
  if [[ ${next_idx} -lt ${#models[@]} ]]; then
    next_model="${models[$next_idx]}"
  fi

  model_out="${OUT_ROOT}/${model}"
  model_vec="${VEC_ROOT}/${model}"
  model_log_dir="${LOG_ROOT}/${model}"
  mkdir -p "${model_out}" "${model_vec}" "${model_log_dir}"

  middle1="$(py -c 'import json,sys; d=json.load(open(sys.argv[1])); print(",".join(str(x) for x in d[sys.argv[2]]["middle1"]))' "${PLAN_JSON}" "${model}")"
  middle6="$(py -c 'import json,sys; d=json.load(open(sys.argv[1])); print(",".join(str(x) for x in d[sys.argv[2]]["middle6"]))' "${PLAN_JSON}" "${model}")"

  if [[ -n "${next_model}" ]]; then
    next_model_id="$(py -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d[sys.argv[2]]["model_id"])' "${PLAN_JSON}" "${next_model}")"
    prefetch_log="${LOG_ROOT}/prefetch_${next_model}.log"
    if [[ ${DRY_RUN} -eq 1 ]]; then
      echo "[dry-run] prefetch next model on CPU: ${next_model} (${next_model_id})"
    else
      echo "[prefetch] ${next_model} on CPU"
      CUDA_VISIBLE_DEVICES='' "${PYTHON_BIN}" -c "from transformers import AutoTokenizer,AutoModelForCausalLM; m='${next_model_id}'; AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m); print('prefetch_ok', m)" > "${prefetch_log}" 2>&1 &
    fi
  fi

  base_summary="${model_out}/baseline_summary.tsv"
  base_jsonl="${model_out}/${model}_L${middle1}_a0.jsonl"

  if stage_complete "${base_summary}" "${base_jsonl}"; then
    echo "[skip] baseline complete: ${model}"
  else
    echo "[run] baseline: ${model}"
    run_cmd env MINIWOB_URL="${MINIWOB_URL:-http://localhost:8080/}" CUDA_VISIBLE_DEVICES=0 \
      "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_sweep.py" \
      --model "${model}" \
      --layers "${middle1}" \
      --alphas "0" \
      --base-only \
      --out-dir "${model_out}" \
      --cache-dir "${VEC_ROOT}" \
      --summary-path "${base_summary}" \
      2>&1 | tee "${model_log_dir}/baseline.log"
  fi

  steer_summary="${model_out}/steer_summary.tsv"
  steer_jsonl="${model_out}/${model}_L$(echo "${middle6}" | cut -d',' -f1)_a3.jsonl"

  if stage_complete "${steer_summary}" "${steer_jsonl}"; then
    echo "[skip] steer complete: ${model}"
  else
    echo "[run] steer alpha=3 layers=${middle6}: ${model}"
    run_cmd env MINIWOB_URL="${MINIWOB_URL:-http://localhost:8080/}" CUDA_VISIBLE_DEVICES=0 \
      "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_sweep.py" \
      --model "${model}" \
      --layers "${middle6}" \
      --alphas "3.0" \
      --out-dir "${model_out}" \
      --cache-dir "${VEC_ROOT}" \
      --summary-path "${steer_summary}" \
      2>&1 | tee "${model_log_dir}/steer.log"
  fi
done
