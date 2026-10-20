"""Одинаковые объекты рисуются один раз; лимит просмотра не меняет рендер."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scene_model import (ROOT, read_json, resolve_scene, parameter_values,
                         combinations, sampled_combinations, sweep_count, nodes_by_id)
from scene_variants import collect_preview_instances, instance_key


class VariantTests(unittest.TestCase):
    def setUp(self):
        self.base=read_json(ROOT/'scene.json')
        self.scene=resolve_scene(self.base)
        self.current=parameter_values(self.scene)
        self.sweep={'schema_version':1,'ranges':{
            key:dict(min=value,max=value,step=.01,enabled=False,value=value,preview_full=True)
            for key,value in self.current.items()}}

    def enable(self,key,lo,hi,step):
        self.sweep['ranges'][key].update(min=lo,max=hi,step=step,enabled=True)

    def test_boundaries_default_and_current_with_per_parameter_flags(self):
        for spec in self.sweep['ranges'].values():
            spec.pop('preview_full')
        self.enable('camera_dx_m',-.02,.02,.01)
        self.enable('camera_rz_deg',-2,2,1)
        instances,shown,total=collect_preview_instances(self.scene,self.sweep)
        self.assertEqual((shown,total),(4,25))
        cameras=[i for i in instances if i.node['type']=='camera']
        self.assertEqual(len(cameras),5)  # Четыре угла диапазонов + текущая камера.
        self.sweep['ranges']['camera_dx_m']['preview_full']=True
        instances,shown,total=collect_preview_instances(self.scene,self.sweep)
        self.assertEqual((shown,total),(10,25))  # 5 сдвигов × 2 крайних угла.
        self.sweep['ranges']['camera_rz_deg']['preview_full']=True
        instances,shown,total=collect_preview_instances(self.scene,self.sweep)
        self.assertEqual((shown,total),(25,25))
        self.assertEqual(sum(i.node['type']=='camera' for i in instances),25)
        self.assertEqual(sweep_count(self.sweep,self.current),25)

    def test_boundaries_use_last_actual_grid_value_and_preview_limit(self):
        for spec in self.sweep['ranges'].values():
            spec['preview_full']=False
        self.enable('camera_dx_m',0,.025,.01)
        rows,total=sampled_combinations(self.sweep,self.current,10)
        self.assertEqual([v['camera_dx_m'] for v in rows],[0,.02])
        self.assertEqual(total,3)
        self.enable('camera_dy_m',0,.02,.01)
        rows,total=sampled_combinations(self.sweep,self.current,2)
        self.assertEqual(len(rows),2)
        self.assertEqual(total,9)

    def test_legacy_global_flag_is_fallback_only(self):
        self.enable('camera_dx_m',-.02,.02,.01)
        self.sweep['ranges']['camera_dx_m'].pop('preview_full')
        self.scene['preview']['full_display']=True
        self.assertEqual(collect_preview_instances(self.scene,self.sweep)[1],5)
        self.sweep['ranges']['camera_dx_m']['preview_full']=False
        self.assertEqual(collect_preview_instances(self.scene,self.sweep)[1],2)

    def test_disabled_uses_current_and_preserves_range(self):
        self.sweep['ranges']['yaw_deg'].update(min=-50,max=50,step=1,value=999)
        self.current['yaw_deg']=7
        rows=list(combinations(self.sweep,self.current))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['yaw_deg'],7)
        self.assertEqual(self.sweep['ranges']['yaw_deg']['min'],-50)

    def test_offsets_move_only_camera_and_merge_other_objects(self):
        self.enable('camera_dx_m',-.01,.01,.01)
        instances,shown,total=collect_preview_instances(self.scene,self.sweep)
        self.assertEqual((shown,total),(3,3))
        for kind,count in [('camera',3),('spherical_shell',1),('checkerboard',1),('group',1)]:
            self.assertEqual(sum(i.node['type']==kind for i in instances),count)
        for item in instances:
            if item.node['type']!='camera':
                self.assertEqual(item.variants,[1,2,3])
        offset=resolve_scene(self.scene,{'camera_dx_m':.01,'camera_rz_deg':2})
        self.assertEqual(nodes_by_id(offset)['board'],nodes_by_id(self.scene)['board'])
        self.assertNotEqual(nodes_by_id(offset)['camera'],nodes_by_id(self.scene)['camera'])

    def test_ior_creates_no_duplicate_visible_objects(self):
        self.enable('ior',1.3,1.7,.1)
        instances,shown,total=collect_preview_instances(self.scene,self.sweep)
        self.assertEqual((shown,total),(5,5))
        self.assertEqual(len(instances),4)
        self.assertTrue(all(len(i.variants)==5 for i in instances))

    def test_parent_pose_geometry_and_tolerance_in_identity(self):
        node=nodes_by_id(self.scene)['board']
        matrix=np.eye(4)
        key=instance_key(self.scene,node,matrix)
        slightly=matrix.copy(); slightly[0,3]=-1e-12
        self.assertEqual(key,instance_key(self.scene,node,slightly))
        moved=matrix.copy(); moved[0,3]=.001
        self.assertNotEqual(key,instance_key(self.scene,node,moved))
        changed=deepcopy(node);changed['geometry']['rows']+=1
        self.assertNotEqual(key,instance_key(self.scene,changed,matrix))

    def test_sampling_huge_product_without_full_enumeration(self):
        for key in self.current:
            self.enable(key,0,9999,1)
        rows,total=sampled_combinations(self.sweep,self.current,7)
        self.assertEqual(total,10000**len(self.current))
        self.assertEqual(len(rows),7)
        self.assertTrue(all(v==0 for v in rows[0].values()))
        self.assertTrue(all(v==9999 for v in rows[-1].values()))
        self.assertEqual(sweep_count(self.sweep,self.current),total)

    def test_current_pose_included_outside_sample(self):
        self.enable('camera_dx_m',.1,.2,.1)
        instances,shown,total=collect_preview_instances(self.scene,self.sweep,1)
        cameras=[i for i in instances if i.node['type']=='camera']
        self.assertEqual(len(cameras),2)
        self.assertEqual(sum(i.current for i in cameras),1)
        self.assertEqual((shown,total),(1,2))

    def test_multiple_geometry_and_full_frustums(self):
        from scene_preview import build_multiple_preview
        self.enable('camera_dx_m',-.01,.01,.01)
        objects,message=build_multiple_preview(self.scene,self.sweep)
        self.assertEqual(len(objects[0].children),1)
        self.assertEqual(len(objects[1].children),1)
        self.assertEqual(len(objects[2].children),3)
        self.assertIn('3 из 3',message)
        self.scene['preview']['variant_frustums']=True
        full,_=build_multiple_preview(self.scene,self.sweep)
        self.assertEqual(len(full[2].children),3)

    def test_viewer_orbit_target_uses_real_camera_world_position(self):
        from unittest.mock import patch
        from scene_model import Scene
        from scene_preview import Preview
        scene=deepcopy(self.scene)
        scene['graph']['transform']={'translation_m':[1,2,3],'rotation_deg':[10,20,30]}
        scene=resolve_scene(scene,{'camera_dx_m':.02,'camera_dy_m':-.01})
        expected=next(matrix[:3,3]*1000 for obj,matrix in Scene.from_dict(scene).walk()
                      if obj.data['type']=='camera')
        viewer=Preview()
        previous={'target':[0,0,0],'position':[1000,-2000,3000],
                  'quaternion':[0,0,0,1],'zoom':2.5}
        with patch('ocp_vscode.show') as show, patch('scene_preview.build_preview',return_value=[]), patch('ocp_vscode.status',return_value=previous):
            viewer.show(scene)
            np.testing.assert_allclose(show.call_args.kwargs['target'],expected)
            np.testing.assert_allclose(show.call_args.kwargs['position'],expected+[1000,-2000,3000])
            self.assertEqual(show.call_args.kwargs['zoom'],2.5)
            viewer.show(resolve_scene(scene,{'ior':1.4}))
            self.assertNotIn('target',show.call_args.kwargs)
            moved=resolve_scene(scene,{'camera_dx_m':.03})
            viewer.show(moved)
            self.assertIn('target',show.call_args.kwargs)
            self.assertFalse(np.allclose(show.call_args.kwargs['target'],expected))
            np.testing.assert_allclose(np.array(show.call_args.kwargs['position'])-show.call_args.kwargs['target'],[1000,-2000,3000])
        from scene_preview import orbit_camera_options
        initial=orbit_camera_options(scene,expected,{})
        self.assertIn('position',initial)
        self.assertGreater(np.linalg.norm(np.array(initial['position'])-expected),0)

    def test_real_batch_uses_disabled_and_camera_offset_parameters(self):
        from scene_export import render_batch
        self.scene['render'].update(resolution_scale=.0125,samples=1)
        self.enable('camera_dy_m',0,.01,.01)
        self.sweep['ranges']['yaw_deg'].update(min=20,max=30,step=5)
        with tempfile.TemporaryDirectory() as directory:
            render_batch(self.scene,self.sweep,directory)
            frames=[read_json(Path(directory)/f'{i:06d}'/'scene.json') for i in [1,2]]
            self.assertEqual([f['parameters']['camera_dy_m'] for f in frames],[0,.01])
            self.assertEqual([f['parameters']['yaw_deg'] for f in frames],[0,0])
            self.assertEqual(nodes_by_id(frames[0])['board'],nodes_by_id(frames[1])['board'])
