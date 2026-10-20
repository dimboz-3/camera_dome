#!/usr/bin/env bash
# Набор сцен → рендер → calibmar → CSV.
# Запуск из любого каталога:
#   bash /home/dfrolov/workspace/3d_scene/run_camera_sweep.sh
# Только подготовить JSON и показать число кадров (без расчёта):
#   bash /home/dfrolov/workspace/3d_scene/run_camera_sweep.sh --prepare-only
# Необязательный последний аргумент — папка результата.
set -euo pipefail

# РЕДАКТИРОВАТЬ ЗДЕСЬ: "от до шаг". При от=до параметр фиксирован.
# Переносы — МЕТРЫ, повороты — ГРАДУСЫ, локальные оси реальной камеры.
# Это camera_d* / camera_r*, а не pitch/yaw/roll опорной камеры.
DX="${DX:--0.01 0.01 0.01}"    # -1 см, 0, +1 см: 3 значения
DY="${DY:-0 0 0.01}"
DZ="${DZ:-0 0 0.01}"
RX="${RX:-0 0 1}"
RY="${RY:--2 2 2}"             # -2°, 0°, +2°: 3 значения
RZ="${RZ:-0 0 1}"
# По умолчанию 3 × 3 = 9 кадров. Все включённые диапазоны перемножаются!
BOARD_DISTANCE_M="${BOARD_DISTANCE_M:-1}"
WIDTH="${WIDTH:-1920}"         # Для текущей камеры высота получится 1536.
SAMPLES="${SAMPLES:-64}"
VARIANT="${VARIANT:-cuda_ad_rgb}"  # CPU: scalar_rgb или llvm_ad_rgb.

scene_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Сейчас общее окружение проекта находится в домашнем каталоге.
# Для другого расположения: PIXI_MANIFEST=/путь/pixi.toml bash ...
pixi_manifest="${PIXI_MANIFEST:-$HOME/pixi.toml}"
source_scene="${SOURCE_SCENE:-$scene_dir/scene.json}"
prepare_only=false
if [[ "${1:-}" == --prepare-only ]]; then prepare_only=true; shift; fi
[[ $# -le 1 ]] || { echo 'Использование: run_camera_sweep.sh [--prepare-only] [ПАПКА]' >&2; exit 2; }
output="${1:-$scene_dir/renders/camera_sweep_$(date +%Y%m%d_%H%M%S_%N)}"
# Получаем абсолютный путь до смены рабочего каталога в дочерних процессах.
output=$(realpath -m -- "$output")

# JSON готовим средствами Python из того же окружения, что использует рендерер.
# Основные scene.json/sweep.json и состояние панели не меняются.
pixi run --manifest-path "$pixi_manifest" python - "$scene_dir" "$source_scene" "$output" \
    "$BOARD_DISTANCE_M" "$DX" "$DY" "$DZ" "$RX" "$RY" "$RZ" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scene_model import read_json, write_json, resolve_scene, parameter_values, sweep_count, sweep_axes

source, output = Path(sys.argv[2]), Path(sys.argv[3])
scene = resolve_scene(read_json(source), {'board_distance_m': float(sys.argv[4])})
values = parameter_values(scene)
# Остальные параметры фиксируем на значениях исходной сцены.
sweep = {'schema_version': 1, 'ranges': {
    key: {'min': value, 'max': value, 'step': 1, 'enabled': False}
    for key, value in values.items()
}}
for key, text in zip(('camera_dx_m', 'camera_dy_m', 'camera_dz_m',
                       'camera_rx_deg', 'camera_ry_deg', 'camera_rz_deg'), sys.argv[5:]):
    start, stop, step = map(float, text.split())
    sweep['ranges'][key] = {'min': start, 'max': stop, 'step': step, 'enabled': True}
# Проверяем диапазоны ДО записи файлов и начала длительного расчёта.
axes = sweep_axes(sweep, values)
count = sweep_count(sweep, values)
# Новая папка исключает случайную замену существующего эксперимента.
output.mkdir(parents=True, exist_ok=False)
write_json(output/'input_scene.json', scene)
write_json(output/'input_sweep.json', sweep)
print(f'Папка: {output}\nКадров: {count}', flush=True)
for key in ('camera_dx_m', 'camera_dy_m', 'camera_dz_m',
                                     'camera_rx_deg', 'camera_ry_deg', 'camera_rz_deg'):
    print(f'  {key}: {axes[key]}', flush=True)
PY

if "$prepare_only"; then
    echo 'Подготовка завершена. Рендер и калибровка не запускались.'
    exit 0
fi

# 1. Генерация всех сочетаний и рендер. Светящаяся доска, как в последних тестах.
render_status=0
pixi run --manifest-path "$pixi_manifest" python "$scene_dir/scene_export.py" \
    --scene "$output/input_scene.json" --sweep "$output/input_sweep.json" \
    --out "$output" --width "$WIDTH" --samples "$SAMPLES" \
    --variant "$VARIANT" --board-mode emissive || render_status=$?
printf 'render=%s\n' "$render_status" > "$output/pipeline_status.txt"

# 2. Независимая калибровка каждого кадра. Сбор CSV нужен и при сбое части кадров.
calibration_status=0
bash "$scene_dir/run_calibmar.sh" "$output" || calibration_status=$?

# 3. Таблица результатов для Excel/Jupyter.
table_status=0
pixi run --manifest-path "$pixi_manifest" python "$scene_dir/collect_calibmar_results.py" \
    "$output" || table_status=$?
printf 'calibration=%s\ntable=%s\n' "$calibration_status" "$table_status" >> "$output/pipeline_status.txt"
echo "Таблица: $output/analysis/results.csv"
[[ "$render_status" == 0 && "$calibration_status" == 0 && "$table_status" == 0 ]]
