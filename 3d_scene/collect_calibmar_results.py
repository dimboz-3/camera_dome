#!/usr/bin/env python3
"""Собрать готовые ответы calibmar в CSV. Ничего не запускает и не пересчитывает.

    python 3d_scene/collect_calibmar_results.py /путь/к/набору

Читает scene.json, dome_position.json (или старый ground_truth.json) и calibmar_result/image.png/calibmar_result.json.
Пишет ПАПКА_НАБОРА/analysis/results.csv (UTF-8 с BOM, разделитель ; для Excel).
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys


def collect_case(case, root):
    """Превратить один ответ в строку таблицы; недостающие метрики оставить пустыми."""
    row = {'case_id': case.relative_to(root).as_posix(), 'status': 'ERROR', 'message': ''}
    try:
        scene = json.loads((case/'scene.json').read_text(encoding='utf-8'))
        truth_path = case/'dome_position.json'
        if not truth_path.exists():
            truth_path = case/'ground_truth.json'  # старые наборы
        truth = json.loads(truth_path.read_text(encoding='utf-8'))
        row.update({'param.'+key: value for key, value in scene.get('parameters', {}).items()})
        row.update({'render.'+key: scene.get('render', {}).get(key, '')
                    for key in ('seed', 'samples', 'variant', 'board_mode')})
        row.update(inner_radius_input_m=truth['dome_inner_radius_m'],
                   thickness_input_m=truth['dome_thickness_m'], glass_ior_input=truth['glass_ior'])
        if truth.get('length_unit') != 'm':
            raise ValueError('Ожидаются длины в метрах')
        image_name = truth.get('image_file', 'image.png')
        if Path(image_name).name != image_name:
            raise ValueError('image_file должен быть именем файла')
        frame = case/'calibmar_result'/image_name
        result_path = frame/'calibmar_result.json'
        row.update(result_json=str(result_path), overlay_png=str(frame/'overlay.png'))

        # У запуска может не быть JSON ответа: например, CLI не смог прочитать кадр.
        # Новый run.json также не даёт принять старый ответ за успешный новый запуск.
        # Bash сохраняет код завершения в обычный текстовый файл.
        exit_path = case/'calibmar_result/exit_code.txt'
        if exit_path.exists():
            code = exit_path.read_text().strip()
            if code != '0':
                row.update(status='RUNNING' if code == 'RUNNING' else 'CLI_ERROR',
                           message=f'CLI exit {code}; см. calibmar_result/calibmar.log')
                return row
        # Поддержка результатов предыдущей Python-обёртки.
        run_path = case/'calibmar_result/run.json'
        if run_path.exists():
            run = json.loads(run_path.read_text(encoding='utf-8'))
            if run.get('status') and run['status'] != 'OK':
                row.update(status=run['status'], message=run.get('message', ''))
                return row
        if not result_path.exists():
            row.update(status='NOT_PROCESSED', message='Нет JSON результата')
            return row
        result = json.loads(result_path.read_text(encoding='utf-8'))
        row.update(status=result['status'], message=result.get('message', ''),
                   visualization_status=result.get('visualization_status', ''))
        if result['status'] != 'OK':
            return row

        # RMSE уже посчитан CLI по найденным узлам, в пикселях исходного кадра.
        # Это не ошибка детекции относительно идеальной проекции.
        for key in ('num_points', 'num_rejected', 'max_error_px', 'max_error_point_index', 'rmse_px', 'rmse_all_points_px', 'rmse_inliers_px', 'fit_rmse_px'):
            value = result.get(key, '')
            if value != '' and not math.isfinite(value):
                raise ValueError(f'Неконечная метрика {key}')
            row[key] = value
        expected = truth['dome_center_camera_opencv_m']
        # Новый отчёт содержит именованный объект; старые плоские поля тоже читаем.
        housing = result.get('housing_estimated', {})
        estimated = housing.get('dome_center_camera_opencv_m', result.get('dome_center_camera_opencv_m'))
        if estimated is None and 'housing_estimated_params' in result:
            estimated = result['housing_estimated_params'][:3]
        if len(expected) != 3 or len(estimated) != 3 or not all(math.isfinite(v) for v in expected+estimated):
            raise ValueError('Ожидаются два конечных трёхмерных центра')

        # Обе позиции — метры, OpenCV: X вправо, Y вниз, Z вперёд.
        # Сохраняем сами центры, знаковые ошибки по осям и длину вектора ошибки.
        error = [a-b for a, b in zip(estimated, expected)]
        for axis, actual, fitted, delta in zip('xyz', expected, estimated, error):
            row[f'center_true_{axis}_m'] = actual
            row[f'center_estimated_{axis}_m'] = fitted
            row[f'center_error_{axis}_m'] = delta
        row['center_error_norm_m'] = math.sqrt(sum(v*v for v in error))
    except (OSError, ValueError, KeyError, TypeError) as error:
        row.update(status='ERROR', message=str(error))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--csv', type=Path, help='По умолчанию DIRECTORY/analysis/results.csv')
    parser.add_argument('--delimiter', choices=[';', ','], default=';')
    args = parser.parse_args(argv)
    root = args.directory.resolve()
    if not root.is_dir():
        parser.error('Нет указанной папки')
    cases = sorted({path.parent for name in ('dome_position.json', 'ground_truth.json')
                    for path in root.rglob(name)})
    if not cases:
        parser.error('Не найдено папок с dome_position.json или ground_truth.json')
    rows = [collect_case(case, root) for case in cases]

    # Объединение ключей сохраняет все параметры даже при разных наборах столбцов.
    # Пустое поле означает отсутствие результата; нулём его не подменяем.
    leading = ['case_id', 'status', 'message', 'rmse_px', 'max_error_px', 'center_error_norm_m']
    columns = leading + sorted(set().union(*(row.keys() for row in rows)) - set(leading))
    output = (args.csv or root/'analysis/results.csv').resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter=args.delimiter)
        writer.writeheader()
        writer.writerows(rows)
    failures = sum(row['status'] != 'OK' for row in rows)
    print(f'CSV: {output}\nСтрок: {len(rows)}, без успешного результата: {failures}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
