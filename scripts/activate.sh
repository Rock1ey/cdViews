#!/usr/bin/env bash
# Source this file: source scripts/activate.sh lab|cloud [GPU_INDEX]

_cdv_activate() {
    local profile="${1:-}" gpu="${2:-0}" repo env_prefix
    repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || return
    case "$profile" in
        lab)
            env_prefix=/home/pengjiale/miniforge3/envs/cdViews
            export CDV_DATA=/home/pengjiale/datasets/qa
            ;;
        cloud)
            env_prefix=/workspace/cdviews-work/envs/cdViews
            export CDV_DATA=/workspace/cdviews-work/qa
            ;;
        *) printf 'Usage: source scripts/activate.sh lab|cloud [GPU_INDEX]\n' >&2; return 1 ;;
    esac
    export CDV_REPO="$repo" CDV_HOST_ID="$profile" CDV_GPU="$gpu"
    export CDV_ENV_PREFIX="$env_prefix"
    # Optional private overrides; never commit credentials or machine-specific edits.
    if [[ -f "$repo/scripts/machine.local.sh" ]]; then
        source "$repo/scripts/machine.local.sh" || return
    fi
    if [[ ! -x "$CDV_ENV_PREFIX/bin/python" ]]; then
        printf 'Missing environment: %s\n' "$CDV_ENV_PREFIX" >&2
        return 1
    fi
    if [[ "$profile" == lab ]]; then
        source /home/pengjiale/miniforge3/etc/profile.d/conda.sh || return
        conda activate "$CDV_ENV_PREFIX" || return
    else
        source "$CDV_ENV_PREFIX/bin/activate" || return
    fi
    export HF_HOME="$CDV_DATA/hf_cache"
    export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
    export CDV_RUNS="$CDV_DATA/runs/$CDV_HOST_ID"
    unset TRANSFORMERS_CACHE
    printf 'Host=%s GPU=%s\nRepo=%s\nData=%s\nPython=%s\n' \
        "$CDV_HOST_ID" "$CDV_GPU" "$CDV_REPO" "$CDV_DATA" "$(command -v python)"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Use source scripts/activate.sh lab|cloud [GPU_INDEX]\n' >&2
    exit 1
fi
_cdv_activate "$@"
_cdv_status=$?
unset -f _cdv_activate
return "$_cdv_status"
