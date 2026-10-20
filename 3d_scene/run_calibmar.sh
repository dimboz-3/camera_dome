#!/usr/bin/env bash
# Запуск: bash 3d_scene/run_calibmar.sh ПАПКА_НАБОРА [ПУТЬ_К_CLI]
# В каждой папке кадра уже должен лежать calibmar_config.json от экспортёра.
# Для CSV затем отдельно запустите collect_calibmar_results.py.
set -uo pipefail

root=$(realpath -e -- "${1:?Укажи папку набора}") || exit 2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 2
cli=$(realpath -e -- "${2:-$script_dir/../calibmar/build/src/cli/calibmar_cli}") || exit 2
[[ -d "$root" && -x "$cli" ]] || { echo 'Нужны папка набора и исполняемый CLI' >&2; exit 2; }

# Настройки просмотра/времени можно задать перед командой: UPSAMPLE=4 TIMEOUT_SEC=300 bash ...
upsample=${UPSAMPLE:-4}
timeout_sec=${TIMEOUT_SEC:-300}
[[ "$upsample" =~ ^([1-9]|1[0-6])$ && "$timeout_sec" =~ ^[1-9][0-9]*$ ]] || {
    echo 'UPSAMPLE: целое 1–16; TIMEOUT_SEC: положительное целое' >&2; exit 2;
}

# Только процесс этого скрипта и его потомки получают библиотеки Pixi.
# Для нестандартной установки можно заранее задать LD_LIBRARY_PATH.
lib_dir="$(dirname -- "$cli")/../../../.pixi/envs/default/lib"
[[ ! -d "$lib_dir" ]] || export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$cli" --help >/dev/null || exit 2

count=0
failures=0
# -print0 сохраняет пробелы в именах папок. Вложенные служебные папки пропускаем.
while IFS= read -r -d '' config; do
    folder=$(dirname -- "$config")
    output="$folder/calibmar_result"
    mkdir -p -- "$output" || exit 2
    count=$((count + 1))

    # Не оставляем успешные файлы прошлого запуска рядом с новым неудачным.
    find "$output" -mindepth 2 -maxdepth 2 -type f \( \
        -name 'calibmar_result.json' -o -name 'calibmar_result.yaml' -o \
        -name 'calibmar_result.txt' -o -name 'overlay.png' -o -name 'image_upsampled.png' \) -delete || exit 2
    rm -f -- "$output/run.json"  # статус прежней Python-обёртки, если она использовалась
    printf 'RUNNING\n' > "$output/exit_code.txt"

    # Именно cd делает images.directory="." переносимым вместе с папкой сцены.
    # Ненулевой код обрабатываем сами, чтобы перейти к следующему кадру.
    (
        cd -- "$folder" || exit 2
        timeout "$timeout_sec" "$cli" calibrate --config calibmar_config.json \
            --per-image --output-dir calibmar_result --upsample "$upsample"
    ) > "$output/calibmar.log" 2>&1
    code=$?
    printf '%s\n' "$code" > "$output/exit_code.txt"
    # Для больших наборов можно не хранить копию исходника: KEEP_UPSAMPLED=0.
    # Сам исходный рендер и overlay остаются на месте.
    if [[ "${KEEP_UPSAMPLED:-1}" == 0 ]]; then
        find "$output" -mindepth 2 -maxdepth 2 -type f -name image_upsampled.png -delete
    fi
    printf '%s: exit %s (журнал: %s/calibmar.log)\n' "$folder" "$code" "$output"
    [[ "$code" == 0 ]] || failures=$((failures + 1))
done < <(find "$root" -type d \( -name calibmar_result -o -name analysis \) -prune -o \
    -type f -name calibmar_config.json -print0)

[[ "$count" != 0 ]] || { echo 'Не найдено calibmar_config.json; нужен экспорт сцены новым экспортёром' >&2; exit 2; }
printf 'Обработано: %s, ошибок: %s\n' "$count" "$failures"
[[ "$failures" == 0 ]]
