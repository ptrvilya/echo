#!/bin/bash
# Full preprocessing of all datasets into env.datasets_folder.
#
# Usage:
#   scripts/preprocess.sh                      # amass behave omomo
#   DATASETS="behave omomo" scripts/preprocess.sh
#   CLEAN=1 scripts/preprocess.sh              # delete previous outputs first
#   BEHAVE_CONVERT=1 scripts/preprocess.sh     # also rerun SMPL-H -> SMPL-X fit
#   DRY_RUN=1 scripts/preprocess.sh            # print the commands only
#
# Roots of the raw datasets come from the env config (CONFIG, default
# scenarios/env.yaml: env.amass_folder, env.behave_folder, ...) and can
# be overridden through EXTRA_OVERRIDES (e.g. "env.amass_folder=/data/amass").
# Canonical object transforms, keypoints and pointnext features are taken
# from assets/objects/<dataset> (common.objects_assets_folder); only the
# canonical meshes are regenerated from the raw datasets.
set -euo pipefail

cd "$(dirname "$0")/.."

DATASETS="${DATASETS:-amass behave omomo}"
DATASETS_FOLDER="${DATASETS_FOLDER:-/mnt/nvme1/echo}"
CLEAN="${CLEAN:-0}"
BEHAVE_CONVERT="${BEHAVE_CONVERT:-0}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
CONFIG="${CONFIG:-scenarios/env.yaml}"
LOG_DIR="${LOG_DIR:-${DATASETS_FOLDER}/preprocess_logs}"

mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/preprocess_$(date +%Y%m%d_%H%M%S).log"

run() {
    local name="$1"
    shift
    echo "[$(date +%H:%M:%S)] ${name}: $*" | tee -a "${LOG}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        return
    fi
    local start
    start=$(date +%s)
    "$@" 2>&1 | tee -a "${LOG}"
    echo "[$(date +%H:%M:%S)] ${name}: done in $(( $(date +%s) - start ))s" | tee -a "${LOG}"
}

parser() {
    # parser <module> [args...] ; env.datasets_folder and EXTRA_OVERRIDES are appended
    local module="$1"
    shift
    local flags=() overrides=()
    for arg in "$@"; do
        if [[ "${arg}" == -* ]]; then flags+=("${arg}"); else overrides+=("${arg}"); fi
    done
    python -m "echo.data_parsers.${module}" -c "${CONFIG}" ${flags[@]+"${flags[@]}"} -- \
        "env.datasets_folder=${DATASETS_FOLDER}" ${EXTRA_OVERRIDES} ${overrides[@]+"${overrides[@]}"}
}

clean() {
    local folder="${DATASETS_FOLDER}/$1"
    [[ "${CLEAN}" == "1" && -d "${folder}" ]] || return 0
    echo "Cleaning ${folder}" | tee -a "${LOG}"
    local targets=(
        "${folder}"/dataset_*_30fps
        "${folder}"/skipped_files_*.txt
        "${folder}/object_keypoints"
        "${folder}/object_meshes"
        "${folder}/canonical_object_transform.pkl"
        "${folder}/object_pointnext.pkl"
    )
    for t in "${targets[@]}"; do
        [[ -e "${t}" ]] || continue
        if [[ "${DRY_RUN}" == "1" ]]; then echo "  rm -rf ${t}"; else rm -rf "${t}"; fi
    done
}

for dataset in ${DATASETS}; do
    echo "===== ${dataset} =====" | tee -a "${LOG}"
    clean "${dataset}"
    case "${dataset}" in
        amass)
            for split in train val test; do
                run "amass/${split}" parser preprocess_amass_smplx "amass.split=${split}"
            done
            ;;
        behave)
            if [[ "${BEHAVE_CONVERT}" == "1" ]]; then
                run "behave/smplh2x" parser convert_behave_smplh2x
            fi
            run "behave/train" parser preprocess_behave -g "behave.split=train"
            run "behave/test" parser preprocess_behave "behave.split=test"
            ;;
        omomo)
            run "omomo/train" parser preprocess_omomo -g "omomo.split=train"
            run "omomo/test" parser preprocess_omomo "omomo.split=test"
            ;;
        adt)
            run "adt/train" parser preprocess_adt -g "adt.split=train"
            run "adt/test" parser preprocess_adt "adt.split=test"
            ;;
        intercap)
            run "intercap/train" parser preprocess_intercap -g "intercap.split=train"
            run "intercap/test" parser preprocess_intercap "intercap.split=test"
            ;;
        *)
            echo "Unknown dataset: ${dataset}" >&2
            exit 1
            ;;
    esac
done

echo "Finished. Log: ${LOG}" | tee -a "${LOG}"
