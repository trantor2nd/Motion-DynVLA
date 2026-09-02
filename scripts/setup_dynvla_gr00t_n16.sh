#!/usr/bin/env bash
set -euo pipefail

project_root="${MOTION_DYNVLA_ROOT:-/mnt/hdd/hesibo/motion_dynvla}"
repo_dir="${project_root}/code/DynVLA-GR00T"
bootstrap_python="${project_root}/envs/dynvla/bin/python"
uv_site="${project_root}/tools/uv_site"
uv_wheel="${project_root}/tools/wheels/uv-0.12.6-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
target_env="${project_root}/envs/gr00t_n16"
uv_cache="${project_root}/cache/uv"
log_dir="${project_root}/logs"

mkdir -p "${uv_site}" "${uv_cache}" "${log_dir}"

if ! PYTHONPATH="${uv_site}" "${bootstrap_python}" -m uv --version >/dev/null 2>&1; then
    "${bootstrap_python}" -m pip install \
        --no-index \
        --no-deps \
        --target "${uv_site}" \
        "${uv_wheel}"
fi

proxy_args=()
if [[ -n "${MOTION_DYNVLA_PROXY:-}" ]]; then
    export HTTP_PROXY="${MOTION_DYNVLA_PROXY}"
    export HTTPS_PROXY="${MOTION_DYNVLA_PROXY}"
    proxy_args+=("proxy=${MOTION_DYNVLA_PROXY}")
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${log_dir}/setup_gr00t_n16_${timestamp}.log"
printf 'repo=%s\nenv=%s\nflash_attn_install=official_wheel_only\n' \
    "${repo_dir}" "${target_env}" | tee "${log_file}"
if ((${#proxy_args[@]})); then
    printf '%s\n' "${proxy_args[@]}" | tee -a "${log_file}"
fi

cd "${repo_dir}"
PYTHONPATH="${uv_site}" \
UV_CACHE_DIR="${uv_cache}" \
UV_PROJECT_ENVIRONMENT="${target_env}" \
"${bootstrap_python}" -m uv sync \
    --frozen \
    --python "${bootstrap_python}" \
    --no-build-package flash-attn 2>&1 | tee -a "${log_file}"

"${target_env}/bin/python" "${repo_dir}/scripts/verify_dynvla_gr00t_n16_env.py" \
    2>&1 | tee -a "${log_file}"
