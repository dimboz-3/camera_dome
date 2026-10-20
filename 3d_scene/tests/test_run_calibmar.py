"""Переносимый конфиг, Bash-запуск и независимый сборщик CSV."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene_model import ROOT, read_json, resolve_scene
from copy import deepcopy
import numpy as np
from scene_export import export_scene
import collect_calibmar_results as collector


class BatchCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root/'fake cli'
        self.binary.write_text('''#!/usr/bin/env python3
import json, sys
from pathlib import Path
if '--help' in sys.argv: sys.exit(0)
a = sys.argv
config = json.loads(Path(a[a.index('--config')+1]).read_text())
assert config['images']['directory'] == '.'
assert Path(config['image']).is_file()
out = Path(a[a.index('--output-dir')+1])/config['image']
out.mkdir(parents=True, exist_ok=True)
failed = Path.cwd().name == 'bad'
result = {'status': 'DETECTION_ERROR' if failed else 'OK', 'visualization_status': 'OK',
          'dome_center_camera_opencv_m': [1, 2, 3], 'rmse_px': 0.25, 'num_points': 468, 'max_error_px': 0.5, 'max_error_point_index': 17}
(out/'calibmar_result.json').write_text(json.dumps(result))
sys.exit(1 if failed else 0)
''')
        self.binary.chmod(0o700)

    def case(self, name):
        case = self.root/name
        export_scene(resolve_scene(read_json(ROOT/'scene.json')), case)
        (case/'image.png').write_bytes(b'fake input')
        return case

    def launch(self, directory=None):
        return subprocess.run(['bash', str(ROOT/'run_calibmar.sh'), str(directory or self.root),
                               str(self.binary)], capture_output=True, text=True)

    def test_export_config_and_pixel_convention(self):
        case = self.case('one')
        config = read_json(case/'calibmar_config.json')
        truth = read_json(case/'dome_position.json')
        self.assertEqual(config['images']['directory'], '.')
        self.assertNotIn('initial_params', config['housing'])
        self.assertFalse(config['initialization']['estimate_dome_offset'])
        np.testing.assert_allclose(config['initialization']['dome_center_camera_opencv_m'],
                                   truth['dome_center_camera_opencv_m'])
        self.assertEqual(config['housing']['layers'][0]['thickness_m'], truth['dome_thickness_m'])
        self.assertEqual(config['housing']['inner_radius_m'], truth['dome_inner_radius_m'])
        self.assertEqual(config['camera']['params'][2], truth['K_opencv'][0][2]+.5)
        self.assertEqual(config['target']['rows'], 19)
        self.assertEqual(config['target']['cols'], 27)

    def test_reference_start_does_not_follow_real_camera_offset(self):
        # Номинальная камера неподвижна; реальной задаём перенос и поворот.
        from scene_export import calibration_ground_truth, calibmar_config
        base = resolve_scene(read_json(ROOT/'scene.json'), {
            'camera_dx_m': 0, 'camera_dy_m': 0, 'camera_dz_m': 0,
            'camera_rx_deg': 0, 'camera_ry_deg': 0, 'camera_rz_deg': 0})
        moved = resolve_scene(base, {'camera_dx_m': .02, 'camera_dy_m': -.01,
                                     'camera_ry_deg': 2, 'camera_rz_deg': 4})
        nominal = calibmar_config(base, calibration_ground_truth(base))['initialization']
        shifted = calibmar_config(moved, calibration_ground_truth(moved))['initialization']
        np.testing.assert_allclose(nominal['dome_center_camera_opencv_m'], shifted['dome_center_camera_opencv_m'])
        self.assertFalse(np.allclose(shifted['dome_center_camera_opencv_m'],
                                     calibration_ground_truth(moved)['dome_center_camera_opencv_m']))
        # Глобальный перенос/поворот всей установки не меняет относительный старт.
        transformed = deepcopy(moved)
        transformed['graph']['transform'] = {'translation_m': [2, -3, 5], 'rotation_deg': [23, -17, 41]}
        config = calibmar_config(transformed, calibration_ground_truth(transformed))
        np.testing.assert_allclose(config['initialization']['dome_center_camera_opencv_m'],
                                   nominal['dome_center_camera_opencv_m'], atol=1e-12)

    def test_relocation_spaces_and_independent_csv(self):
        original = self.case('original')
        moved = self.root/'moved case'
        shutil.move(original, moved)
        result = self.launch(moved)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertFalse((moved/'analysis/results.csv').exists())
        self.binary.unlink()
        self.assertEqual(collector.main([str(moved)]), 0)
        with (moved/'analysis/results.csv').open(encoding='utf-8-sig') as f:
            row, = list(csv.DictReader(f, delimiter=';'))
        self.assertEqual(row['status'], 'OK')
        self.assertIn('center_error_norm_m', row)
        self.assertEqual(row['max_error_px'], '0.5')
        self.assertEqual(row['max_error_point_index'], '17')

    def test_collector_reads_named_housing_and_legacy_truth_filename(self):
        case = self.case('one')
        self.assertEqual(self.launch(case).returncode, 0)
        result_path = case/'calibmar_result/image.png/calibmar_result.json'
        result = read_json(result_path)
        result['housing_estimated'] = {'dome_center_camera_opencv_m': result.pop('dome_center_camera_opencv_m')}
        result_path.write_text(json.dumps(result))
        new = collector.collect_case(case, self.root)
        self.assertEqual(new['status'], 'OK')
        # Старое имя эталона продолжает читаться без повторного рендера.
        (case/'dome_position.json').rename(case/'ground_truth.json')
        legacy = collector.collect_case(case, self.root)
        self.assertEqual(new['center_error_norm_m'], legacy['center_error_norm_m'])

    def test_resume_legacy_truth_does_not_rerender(self):
        from unittest.mock import patch
        from scene_export import render_batch
        from scene_model import parameter_values
        case = self.case('000001')
        scene = read_json(case/'scene.json')
        values = parameter_values(scene)
        sweep = {'schema_version': 1, 'ranges': {
            key: {'min': value, 'max': value, 'step': 1, 'enabled': False}
            for key, value in values.items()}}
        (case/'dome_position.json').rename(case/'ground_truth.json')
        (case/'calibmar_config.json').unlink()
        (case/'render_info.json').write_text('{}')
        (self.root/'manifest.jsonl').write_text(json.dumps({'index': 1, 'status': 'complete'})+'\n')
        with patch('scene_export.render_scene', side_effect=AssertionError('Unexpected rerender')):
            render_batch(scene, sweep, self.root)
        self.assertTrue((case/'calibmar_config.json').exists())

    def test_failed_case_does_not_stop_next(self):
        bad = self.case('bad')
        good = self.case('good')
        result = self.launch()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(collector.collect_case(bad, self.root)['status'], 'CLI_ERROR')
        self.assertEqual(collector.collect_case(good, self.root)['status'], 'OK')

    def test_failed_rerun_clears_old_answer(self):
        case = self.case('one')
        self.assertEqual(self.launch(case).returncode, 0)
        self.binary.write_text('#!/bin/bash\n[[ "$1" == --help ]] && exit 0\nexit 1\n')
        self.assertEqual(self.launch(case).returncode, 1)
        self.assertFalse((case/'calibmar_result/image.png/calibmar_result.json').exists())
        self.assertNotIn('center_error_norm_m', collector.collect_case(case, self.root))


if __name__ == '__main__':
    unittest.main()
