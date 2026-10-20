"""Model/backend agreement, physical boundaries, serialization, and resumable batches."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scene_model import (ROOT, Scene, Object3D, combinations, mat33_2_euler_deg, nodes_by_id, range_values,
                         read_json, resolve_scene, euler_deg_2_mat33, sweep_count, write_json)
from scene_export import export_scene, render_batch


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.base=read_json(ROOT/'scene.json')
        self.scene=resolve_scene(self.base)

    def test_rotation_axis_order_and_degrees(self):
        # X=90°, затем Z=90°: оси X → Y, Y → Z, Z → X.
        np.testing.assert_allclose(euler_deg_2_mat33([90,0,90]),
                                   [[0,0,1],[1,0,0],[0,1,0]],atol=1e-12)

    def test_euler_roundtrip_including_gimbal_lock(self):
        for angles in [[12,-34,56],[-170,89.99,143],[25,90,40],[25,-90,40]]:
            with self.subTest(angles=angles):
                matrix=euler_deg_2_mat33(angles)
                if abs(angles[1])==90:
                    with self.assertWarnsRegex(UserWarning,'Gimbal lock'):
                        restored=mat33_2_euler_deg(matrix)
                else:
                    restored=mat33_2_euler_deg(matrix)
                # В особых положениях сами углы могут отличаться, ориентация — та же.
                np.testing.assert_allclose(euler_deg_2_mat33(restored),matrix,atol=1e-12)

    def test_decimal_grid_and_cartesian_product(self):
        self.assertEqual(range_values({'min':0,'max':0.3,'step':0.1}),[0,0.1,0.2,0.3])
        self.assertEqual(range_values({'min':0,'max':0.31,'step':0.1}),[0,0.1,0.2,0.3])
        for spec in [{'min':0,'max':1,'step':0},{'min':1,'max':0,'step':1}]:
            with self.assertRaises(ValueError): range_values(spec)
        sweep=read_json(ROOT/'sweep.json')
        self.assertEqual(sweep_count(sweep),len(list(combinations(sweep))))

    def test_orbit_reference_and_fixed_gap(self):
        for thickness in [.02,.05,.1]:
            s=resolve_scene(self.base,{'pitch_deg':30,'yaw_deg':10,'thickness_m':thickness})
            nodes=nodes_by_id(s)
            camera=np.array(nodes['camera']['transform']['translation_m'])
            anchor=np.array(nodes['board_reference']['transform']['translation_m'])
            self.assertAlmostEqual(np.linalg.norm(camera),6-thickness-.05)
            np.testing.assert_allclose(anchor,camera)
            self.assertAlmostEqual(camera[2],anchor[2])
            self.assertAlmostEqual(np.linalg.norm(camera),np.linalg.norm(anchor))
            self.assertEqual(nodes['board']['transform']['translation_m'],[2,0,0])

    def test_camera_play_keeps_reference_and_board_fixed(self):
        self.base['parameterization']['camera_offset'] = {
            'translation_m': [.01, -.02, .03], 'rotation_deg': [2, -3, 4]}
        for mode in ['fixed_world', 'radial', 'board_center']:
            self.base['parameterization']['camera_aim'] = mode
            s = resolve_scene(self.base, {'yaw_deg': 12, 'roll_deg': 7})
            world = {obj.data['id']: m for obj,m in Scene.from_dict(s).walk()}
            ref = world['camera_reference']
            np.testing.assert_allclose(world['board'][:3,3], ref[:3,3] + 2*ref[:3,0])
            np.testing.assert_allclose(world['camera'][:3,3], ref[:3,3] + ref[:3,:3] @ [.01,-.02,.03])
            np.testing.assert_allclose(world['camera'][:3,:3], ref[:3,:3] @ euler_deg_2_mat33([2,-3,4]))
            clean = copy.deepcopy(self.base)
            clean['parameterization'].pop('camera_offset')
            baseline = resolve_scene(clean, {'yaw_deg': 12, 'roll_deg': 7})
            unchanged = {obj.data['id']:m for obj,m in Scene.from_dict(baseline).walk()}
            np.testing.assert_allclose(world['board'], unchanged['board'])
            np.testing.assert_allclose(ref, unchanged['camera_reference'])

    def test_camera_aim_modes_and_roll(self):
        for mode in ['radial','board_center','fixed_world']:
            self.base['parameterization']['camera_aim']=mode
            s=resolve_scene(self.base,{'pitch_deg':30,'yaw_deg':5,'roll_deg':12})
            nodes=nodes_by_id(s); cam=nodes['camera']; r=Object3D(cam).local_matrix()[:3,:3]
            np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)
            if mode=='radial':
                p=np.array(cam['transform']['translation_m'])
                np.testing.assert_allclose(r[:,0],p/np.linalg.norm(p),atol=1e-12)
            elif mode=='board_center':
                world={obj.data['id']:m for obj,m in Scene.from_dict(s).walk()}
                d=world['board'][:3,3]-world['camera'][:3,3]
                np.testing.assert_allclose(r[:,0],d/np.linalg.norm(d),atol=1e-12)
            else:
                self.assertAlmostEqual(r[2,0],0)
        self.assertEqual(self.scene['parameters']['roll_deg'],0)

    def test_nested_parent_transform_and_json_roundtrip(self):
        self.base['graph']['transform']={'translation_m':[1,2,3],'rotation_deg':[10,20,30]}
        s=resolve_scene(self.base)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'scene.json';write_json(path,s);restored=read_json(path)
            self.assertEqual(s,restored)
        worlds={obj.data['id']:m for obj,m in Scene.from_dict(s).walk()}
        local=nodes_by_id(s)['camera']
        expected=Object3D(s['graph']).local_matrix()@Object3D(local).local_matrix()
        np.testing.assert_allclose(worlds['camera'],expected)
        # Mitsuba's sensor center and basis come from the identical world matrix.
        with tempfile.TemporaryDirectory() as tmp:
            xml=ET.parse(export_scene(s,tmp))
            matrix=np.fromstring(xml.find('./sensor/transform/matrix').get('value'),sep=' ').reshape(4,4)
            np.testing.assert_allclose(matrix[:3,3],worlds['camera'][:3,3])
            np.testing.assert_allclose(matrix[:3,2],worlds['camera'][:3,0])
            np.testing.assert_allclose(matrix[:3,1],worlds['camera'][:3,2])
            self.assertAlmostEqual(np.linalg.det(matrix[:3,:3]),1)

    def test_export_boundaries_dimensions_texture_and_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml=ET.parse(export_scene(self.scene,tmp))
            spheres=xml.findall('./shape[@type="sphere"]')
            self.assertEqual([float(s.find('float[@name="radius"]').get('value')) for s in spheres],[6,5.95])
            self.assertEqual([s.find('boolean[@name="flip_normals"]').get('value') for s in spheres],['false','true'])
            from PIL import Image
            tex=np.asarray(Image.open(Path(tmp)/'board_texture.png'))
            self.assertEqual(tex.shape,(19,27,3))
            self.assertGreater(tex[0,0,0],tex[0,1,0])
            mat=np.fromstring(xml.find('./shape[@type="rectangle"]/transform/matrix').get('value'),sep=' ').reshape(4,4)
            self.assertAlmostEqual(np.linalg.norm(mat[:3,0])*2,1.134)
            self.assertAlmostEqual(np.linalg.norm(mat[:3,1])*2,.798)
            self.assertFalse(any('marker' in s.get('id','') for s in xml.findall('./shape')))

    def test_invalid_physical_parameters(self):
        for values in [{'thickness_m':6},{'ior':0.99},{'board_distance_m':-1},{'pitch_deg':90}]:
            with self.assertRaises(ValueError): resolve_scene(self.base,values)

    def test_angular_shell_segment_has_exact_solid_angle_and_reference_anchor(self):
        from scene_preview import section_geometry,build_preview
        outer,inner=6000,5950
        for radial in [(0.,0.,1.),(1.,0.,0.)]:
            part=section_geometry(6,.05,radial,20.)
            self.assertTrue(part.is_valid)
            expected=2*math.pi*(1-math.cos(math.radians(10)))/3*(outer**3-inner**3)
            self.assertAlmostEqual(part.volume/expected,1,places=6)
        scene=resolve_scene(self.base)
        scene['preview'].update(shell_segment=True,section_angle_deg=20)
        before=build_preview(scene)[0].bounding_box()
        moved=resolve_scene(scene,{'camera_dy_m':.2,'camera_rx_deg':15})
        after=build_preview(moved)[0].bounding_box()
        np.testing.assert_allclose(tuple(before.min),tuple(after.min),atol=1e-6)
        np.testing.assert_allclose(tuple(before.max),tuple(after.max),atol=1e-6)
        scene['preview']['shell_segment']=False
        full=build_preview(scene)[0]
        self.assertAlmostEqual(full.volume,4*math.pi/3*(outer**3-inner**3),delta=1)

    def test_oblique_shell_segment_mesh_stays_near_spherical_surfaces(self):
        from scene_preview import section_geometry
        from ocp_tessellate.ocp_utils import bounding_box
        from ocp_tessellate.tessellator import tessellate,compute_quality
        from OCP.BRepTools import BRepTools
        part=section_geometry(6,.05,(1.,0.,1.),20.)
        BRepTools.Clean_s(part.wrapped)
        quality=compute_quality(bounding_box(part.wrapped,optimal=False),.1)
        mesh=tessellate(part.wrapped,'oblique-shell-regression',.1,quality,.2)
        vertices=np.asarray(mesh['vertices']).reshape(-1,3)
        triangles=np.asarray(mesh['triangles']).reshape(-1,3)
        begin=0
        for face,count in zip(part.faces(),mesh['triangles_per_face']):
            count=int(count)
            if face.geom_type.name=='SPHERE':
                points=vertices[triangles[begin:begin+count]]
                radius=float(np.median(np.linalg.norm(points,axis=2)))
                errors=np.abs(np.linalg.norm(points.mean(axis=1),axis=1)-radius)
                # Прежний боковой вырез давал здесь более 40 мм при стенке 50 мм.
                self.assertLess(float(errors.max()),2.)
            begin+=count

    def test_ocp_preview_uses_resolved_camera_and_section_only(self):
        from scene_preview import build_preview
        s=resolve_scene(self.base,{'yaw_deg':3,'roll_deg':7})
        objects=build_preview(s)
        self.assertEqual(len(objects),5)
        shell,board,camera,axes,reference=objects
        self.assertTrue(shell.is_valid)
        self.assertLess(shell.bounding_box().size.Y,12000)
        world={obj.data['id']:m for obj,m in Scene.from_dict(s).walk()}
        marker=camera.children[0]
        # Child bounding boxes are local; the compound location supplies the pose.
        position=camera.location.position
        np.testing.assert_allclose([position.X,position.Y,position.Z],world['camera'][:3,3]*1000,atol=1e-6)
        self.assertAlmostEqual(marker.bounding_box().size.X,20)
        self.assertEqual(nodes_by_id(s)['shell']['geometry']['outer_radius_m'],6)

    def test_object_tree_and_standalone_document(self):
        tree = Scene.from_dict(self.scene)
        camera = next(obj for obj, _ in tree.walk() if obj.data['type'] == 'camera')
        np.testing.assert_allclose(camera.world_matrix(), dict((obj.data['id'], matrix) for obj, matrix in tree.walk())['camera'])
        # A resolved scene needs no experiment parameters or resolver for export.
        document = tree.to_dict()
        for key in ['parameters', 'parameterization', 'resolved']:
            document.pop(key)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'standalone.json'
            Scene.from_dict(document).save(path)
            loaded = Scene.load(path)
            self.assertEqual(document, loaded.to_dict())
            export_scene(loaded.to_dict(), tmp)
        child = camera.to_dict()
        np.testing.assert_allclose(Object3D.from_dict(child).local_matrix(), camera.local_matrix())

    def test_build123d_hierarchy_full_shell_and_step(self):
        from build123d import import_step
        # Nontrivial transforms on both a parent and child must survive CAD export.
        document = copy.deepcopy(self.scene)
        document['graph']['transform'] = {'translation_m': [1,2,3], 'rotation_deg': [12,23,34]}
        tree = Scene.from_dict(document)
        assembly = tree.to_build123d()
        self.assertEqual(assembly.label, 'scene')
        rig = assembly.children[0]
        self.assertEqual(rig.label, 'rig')
        self.assertEqual([part.label for part in rig.children], ['shell','board_reference','camera','camera_reference'])
        shell = rig.children[0].children[0]
        self.assertAlmostEqual(shell.bounding_box().size.X, 12000, places=4)
        self.assertAlmostEqual(shell.volume, 4*math.pi/3*(6000**3 - 5950**3), delta=1)
        self.assertEqual(len(rig.children[1].children[0].children), 27*19)
        with tempfile.TemporaryDirectory() as tmp:
            path = tree.export_step(Path(tmp) / 'scene.step')
            restored = import_step(path)
            self.assertTrue(restored.is_valid)
            self.assertEqual(restored.label, 'scene')
            self.assertEqual(Scene.load(path.with_suffix('.scene.json')).to_dict(), tree.to_dict())
            expected = assembly.bounding_box()
            actual = restored.bounding_box()
            np.testing.assert_allclose(tuple(actual.min), tuple(expected.min), atol=1e-4)
            np.testing.assert_allclose(tuple(actual.max), tuple(expected.max), atol=1e-4)

    def test_real_render_two_frames_resume_and_optical_normals(self):
        import mitsuba as mi
        mi.set_variant('scalar_rgb')
        self.base['render']['resolution_scale']=.025
        self.base['render']['samples']=2
        sweep=read_json(ROOT/'sweep.json')
        for key,value in self.base['parameters'].items():
            sweep['ranges'][key]={'min':value,'max':value,'step':1}
        sweep['ranges']['yaw_deg']={'min':0,'max':1,'step':1}
        with tempfile.TemporaryDirectory() as tmp:
            render_batch(self.base,sweep,tmp)
            first=Path(tmp)/'000001'/'image.png'
            stamp=first.stat().st_mtime_ns
            render_batch(self.base,sweep,tmp)
            self.assertEqual(stamp,first.stat().st_mtime_ns)
            records=[json.loads(line) for line in (Path(tmp)/'manifest.jsonl').read_text().splitlines()]
            self.assertEqual(sum(r['status']=='complete' for r in records),2)
            self.assertTrue((Path(tmp)/'000002'/'image.png').exists())
            loaded=mi.load_file(str(Path(tmp)/'000001'/'scene.xml'))
            hit=loaded.ray_intersect(mi.Ray3f([0,0,0],[1,0,0]))
            self.assertAlmostEqual(hit.p.x,5.95,places=5)
            self.assertLess(hit.n.x,0)  # inner boundary faces the air cavity
            with self.assertRaises(ValueError):
                changed=copy.deepcopy(self.base);changed['render']['seed']=1
                render_batch(changed,sweep,tmp)


if __name__=='__main__':
    unittest.main()
