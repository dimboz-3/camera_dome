"""Ground truth в камерной системе и PNG с чёрным окружением."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scene_model import ROOT,read_json,resolve_scene,Scene
from scene_export import calibration_ground_truth,export_scene,render_scene


class CalibrationExportTests(unittest.TestCase):
    def setUp(self):
        self.scene=resolve_scene(read_json(ROOT/'scene.json'))

    def test_center_axes_and_shared_parent_invariance(self):
        truth=calibration_ground_truth(self.scene)
        camera=next(m for obj,m in Scene.from_dict(self.scene).walk() if obj.data['type']=='camera')
        expected=camera[:3,:3].T @ -camera[:3,3]
        np.testing.assert_allclose(truth['dome_center_camera_native_m'],expected)
        np.testing.assert_allclose(truth['dome_center_camera_opencv_m'],[-expected[1],-expected[2],expected[0]])
        self.assertAlmostEqual(np.linalg.norm(expected),5.9)
        document=deepcopy(self.scene)
        document['graph']['transform']={'translation_m':[2,-3,5],'rotation_deg':[23,-17,41]}
        moved=calibration_ground_truth(document)
        np.testing.assert_allclose(moved['dome_center_world_m'],[2,-3,5],atol=1e-12)
        np.testing.assert_allclose(moved['dome_center_camera_opencv_m'],truth['dome_center_camera_opencv_m'],atol=1e-12)
        shifted=calibration_ground_truth(resolve_scene(self.scene,{'camera_dx_m':.02,'camera_rz_deg':3}))
        self.assertFalse(np.allclose(shifted['dome_center_camera_opencv_m'],truth['dome_center_camera_opencv_m']))

    def test_export_black_environment_and_light_and_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            xml=ET.parse(export_scene(self.scene,directory))
            env=xml.find('./emitter[@type="constant"]/rgb')
            self.assertEqual([float(v) for v in env.get('value').split(',')],[0,0,0])
            self.assertIsNotNone(xml.find('./shape[@id="studio_light"]/emitter[@type="area"]'))
            truth=read_json(Path(directory)/'dome_position.json')
            self.assertEqual(truth['length_unit'],'m')
            self.assertEqual(truth['image_resolution_px'],[640,512])
            self.assertEqual(truth['dome_outer_radius_m'],6)
            self.assertAlmostEqual(truth['dome_inner_radius_m'],5.95)
            self.assertEqual(truth['K_opencv'],[[400,0,319.5],[0,400,255.5],[0,0,1]])
            self.assertEqual(self.scene,resolve_scene(read_json(ROOT/'scene.json')))

    def test_only_png_is_written_and_black_background_is_lit(self):
        scene=deepcopy(self.scene)
        scene['render'].update(resolution_scale=1/12,samples=16)
        with tempfile.TemporaryDirectory() as directory:
            render_scene(scene,directory)
            directory=Path(directory)
            self.assertFalse((directory/'image.exr').exists())
            self.assertFalse((directory/'preview.png').exists())
            a=np.asarray(Image.open(directory/'image.png'))
            self.assertEqual(a.shape,(128,160,3))
            np.testing.assert_array_equal(a[:,:,0],a[:,:,1])
            np.testing.assert_array_equal(a[:,:,1],a[:,:,2])
            for y,x in [(0,0),(0,-1),(-1,0),(-1,-1)]:
                np.testing.assert_array_equal(a[y,x],[0,0,0])
            self.assertGreater(float(a[45:85,50:110].mean()),5)
            info=read_json(directory/'render_info.json')
            self.assertEqual(info['lighting_mode'],'studio')
            self.assertNotIn('exr_encoding',info)

    def test_emissive_board_has_texture_emission_no_lamp_and_correct_side(self):
        scene=deepcopy(self.scene)
        scene['render']['board_mode']='emissive'
        with tempfile.TemporaryDirectory() as directory:
            xml=ET.parse(export_scene(scene,directory))
            board=xml.find('./shape[@id="board"]')
            self.assertIsNotNone(board.find('emitter[@type="area"]/texture[@name="radiance"]'))
            self.assertIsNone(xml.find('./shape[@id="studio_light"]'))
            self.assertEqual(board.find('boolean[@name="flip_normals"]').get('value'),'true')
            self.assertEqual(float(board.find('./bsdf/bsdf/spectrum[@name="reflectance"]').get('value')),0.)
            metadata=read_json(Path(directory)/'dome_position.json')
            self.assertEqual(metadata['dome_center_camera_opencv_m'],calibration_ground_truth(self.scene)['dome_center_camera_opencv_m'])
            other_side=resolve_scene(scene,{'camera_dx_m':3})
            xml=ET.parse(export_scene(other_side,directory))
            self.assertIsNone(xml.find('./shape[@id="board"]/boolean[@name="flip_normals"]'))

    def test_emissive_board_remains_bright_without_external_illumination(self):
        scene=resolve_scene(self.scene,{'ior':1})
        scene['render'].update(board_mode='emissive',resolution_scale=1/6,samples=4,key_light_radiance=0)
        with tempfile.TemporaryDirectory() as directory:
            render_scene(scene,directory)
            a=np.asarray(Image.open(Path(directory)/'image.png'))
            self.assertGreater(int(a[128,160,0]),235)
            np.testing.assert_array_equal(a[:,:,0],a[:,:,1])
            np.testing.assert_array_equal(a[0,0],[0,0,0])
            self.assertEqual(read_json(Path(directory)/'render_info.json')['board_mode'],'emissive')

    def test_intrinsics_match_mitsuba_primary_ray_pixel_centers(self):
        import mitsuba as mi
        mi.set_variant('scalar_rgb')
        with tempfile.TemporaryDirectory() as directory:
            rendered=mi.load_file(str(export_scene(self.scene,directory).resolve()))
            sensor=rendered.sensors()[0]
            truth=calibration_ground_truth(self.scene)
            width,height=truth['image_resolution_px']
            transform=np.asarray(truth['T_camera_opencv_from_world'])
            K=np.asarray(truth['K_opencv'])
            for u,v in [(0,0),(100,200),(639,511)]:
                ray,_=sensor.sample_ray(0.,.5,mi.Point2f((u+.5)/width,(v+.5)/height),mi.Point2f(.5,.5))
                direction=transform[:3,:3] @ np.asarray(ray.d)
                pixel=K @ direction
                np.testing.assert_allclose(pixel[:2]/pixel[2],[u,v],atol=1e-3)
