"""The resolved scene graph -> portable Mitsuba XML -> PNG."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from copy import deepcopy
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from scene_model import (ROOT, combinations, fingerprint, read_json, resolve_scene,
                         sweep_count, validate_scene, Scene, write_json, parameter_values)

EXPORTER_VERSION = '1.6'


def prop(parent, kind, name, value):
    return ET.SubElement(parent, kind, name=name, value=str(value).lower() if isinstance(value, bool) else str(value))


def xml_transform(parent, matrix):
    element = ET.SubElement(parent, 'transform', name='to_world')
    ET.SubElement(element, 'matrix', value=' '.join(f'{float(v):.17g}' for v in matrix.ravel()))


def effective_resolution(scene, sensor):
    scale = scene['render'].get('resolution_scale', 1.0)
    if not 0 < scale <= 4:
        raise ValueError('Масштаб разрешения должен быть >0 и ≤4')
    return [max(1, round(v * scale)) for v in sensor['resolution']]



CAMERA_NATIVE_TO_OPENCV = np.array([
    [0., -1., 0., 0.],
    [0., 0., -1., 0.],
    [1., 0., 0., 0.],
    [0., 0., 0., 1.],
])


def calibration_ground_truth(scene):
    flat=[(obj.data,matrix) for obj,matrix in Scene.from_dict(scene).walk()]
    camera,camera_world=next((n,m) for n,m in flat if n['type']=='camera')
    shell,shell_world=next((n,m) for n,m in flat if n['type']=='spherical_shell')
    camera_from_world=np.linalg.inv(camera_world)
    cv_from_world=CAMERA_NATIVE_TO_OPENCV @ camera_from_world
    center=np.append(shell_world[:3,3],1.)
    sensor=camera['sensor']
    width,height=effective_resolution(scene,sensor)
    focal=sensor['focus_px']*width/sensor['resolution'][0]
    geometry=shell['geometry']
    material=scene['materials'][shell['material']]
    return {
        'schema_version':1, 'length_unit':'m', 'camera_id':camera['id'], 'dome_id':shell['id'],
        'camera_native_axes':{'x':'forward','y':'left','z':'up'},
        'camera_opencv_axes':{'x':'right','y':'down','z':'forward'},
        'matrix_convention':'column vectors; T_destination_from_source @ point_source',
        'dome_center_world_m':center[:3].tolist(),
        'dome_center_camera_native_m':(camera_from_world @ center)[:3].tolist(),
        'dome_center_camera_opencv_m':(cv_from_world @ center)[:3].tolist(),
        'T_world_from_camera_native':camera_world.tolist(),
        'T_camera_native_from_world':camera_from_world.tolist(),
        'T_camera_opencv_from_world':cv_from_world.tolist(),
        'T_camera_opencv_from_dome':(cv_from_world @ shell_world).tolist(),
        'dome_outer_radius_m':geometry['outer_radius_m'],
        'dome_inner_radius_m':geometry['outer_radius_m']-geometry['thickness_m'],
        'dome_thickness_m':geometry['thickness_m'], 'glass_ior':material['ior'],
        'sphere_orientation_observable':False,
        'image_resolution_px':[width,height],
        'K_opencv':[[focal,0,(width-1)/2],[0,focal,(height-1)/2],[0,0,1]],
        'pixel_coordinates':'OpenCV convention: top-left pixel center is (0,0); Mitsuba film sample at (u+0.5,v+0.5)',
        'camera_model':'ideal perspective camera before dome refraction; K does not model the dome',
        'image_file':'image.png',
    }


def calibmar_config(scene, truth):
    """Готовый переносимый конфиг CLI; пути считаются от папки кадра.

    Камера и стекло фиксированы. Старт — центр купола относительно ОПОРНОЙ
    камеры; истинное положение реальной камеры с люфтом не подсказываем.
    """
    flat = [(obj.data, matrix) for obj, matrix in Scene.from_dict(scene).walk()]
    board, = [obj for obj, _ in flat if obj['type'] == 'checkerboard']
    shell, shell_world = next((obj, matrix) for obj, matrix in flat if obj['type'] == 'spherical_shell')
    # Опорная камера — известная номинальная установка БЕЗ camera_d*/camera_r*.
    # Старые сцены до отдельного camera_reference используют ту же опорную
    # позу в board_reference. Истинную позу реальной камеры здесь не читаем.
    binding = scene.get('parameterization', {})
    reference_id = binding.get('camera_reference_id', binding.get('board_reference_id'))
    reference_world = next((matrix for obj, matrix in flat if obj['id'] == reference_id), None)
    if reference_world is None:
        raise ValueError('Для старта calibmar нужна опорная камера в parameterization')
    center_world = np.append(shell_world[:3, 3], 1.)
    initial_center = (CAMERA_NATIVE_TO_OPENCV @ np.linalg.inv(reference_world) @ center_world)[:3]
    medium = scene['materials'][shell['material']]['outside_ior']
    k, g = truth['K_opencv'], board['geometry']
    return {
        # Главная точка: OpenCV (0,0) → COLMAP (0.5,0.5).
        'camera': {'model': 'PINHOLE', 'params': [k[0][0], k[1][1], k[0][2]+.5, k[1][2]+.5]},
        # Слои перечисляются от внутренней поверхности наружу.
        # Сейчас рендерер и решатель реализуют один слой стекла (две границы).
        'housing': {'interface': 'DOMEPORT', 'inner_radius_m': truth['dome_inner_radius_m'],
                    'layers': [{'thickness_m': truth['dome_thickness_m'],
                                'refractive_index': truth['glass_ior']}],
                    'inside_refractive_index': medium, 'outside_refractive_index': medium},
        # Считаем номинальные координаты приближением к координатам реальной
        # камеры. Неизвестный люфт не подсказываем: решатель должен его учесть.
        # Автооценку отключаем, иначе CLI заменит наш старт своим.
        'initialization': {'dome_center_camera_opencv_m': initial_center.tolist(),
                           'estimate_dome_offset': False},
        # Размер доски задан в клетках, не в углах.
        'target': {'rows': g['rows'], 'cols': g['columns'], 'square_size': g['cell_size_m']},
        'image_size': truth['image_resolution_px'],
        'images': {'directory': '.'}, 'image': 'image.png',
        'fast_detection': False, 'detect_padding': 0, 'outlier_rejection_factor': 0,
    }


def add_studio_light(xml, scene, flat):
    camera_pos=next(matrix[:3,3] for node,matrix in flat if node['type']=='camera')
    board,pose=next((node,matrix) for node,matrix in flat if node['type']=='checkerboard')
    geometry=board['geometry']
    size=max(geometry['columns']*geometry['cell_size_m'],geometry['rows']*geometry['cell_size_m'],1.)
    normal=pose[:3,2].copy()
    if np.dot(normal,camera_pos-pose[:3,3])<0:
        normal=-normal
    position=pose[:3,3]+normal*size*1.3+pose[:3,1]*size*2.5
    forward=pose[:3,3]-position
    forward/=np.linalg.norm(forward)
    right=pose[:3,0]
    up=np.cross(forward,right)
    matrix=np.eye(4)
    matrix[:3,0]=right*size*.75
    matrix[:3,1]=up*size*.75
    matrix[:3,2]=forward
    matrix[:3,3]=position
    light=ET.SubElement(xml,'shape',type='rectangle',id='studio_light')
    xml_transform(light,matrix)
    bsdf=ET.SubElement(light,'bsdf',type='diffuse')
    prop(bsdf,'spectrum','reflectance',0)
    emitter=ET.SubElement(light,'emitter',type='area')
    value=scene['render']['key_light_radiance']
    # Равномерный спектр E не является нейтральным sRGB-белым D65.
    prop(emitter,'rgb','radiance',f'{value}, {value}, {value}')

def export_scene(scene, directory):
    """No CAD meshes: analytic boundaries, rectangles, and an actual sensor."""
    scene=deepcopy(scene)
    scene['render'].setdefault('lighting_mode','studio')
    scene['render'].setdefault('background_radiance',0.)
    scene['render'].setdefault('key_light_radiance',12.)
    scene['render'].setdefault('board_mode','diffuse')
    tree = Scene.from_dict(scene)
    scene = tree.to_dict()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'scene.json', scene)
    truth = calibration_ground_truth(scene)
    write_json(directory / 'dome_position.json', truth)
    write_json(directory / 'calibmar_config.json', calibmar_config(scene, truth))
    xml = ET.Element('scene', version='3.0.0')
    integrator = ET.SubElement(xml, 'integrator', type='path')
    prop(integrator, 'integer', 'max_depth', scene['render']['max_depth'])
    env = ET.SubElement(xml, 'emitter', type='constant')
    board_mode=scene['render']['board_mode']
    if board_mode not in ('diffuse','emissive'):
        raise ValueError('Неизвестный режим доски')
    lighting=scene['render']['lighting_mode']
    if lighting not in ('studio','environment'):
        raise ValueError('Неизвестный режим освещения')
    radiance=scene['render']['background_radiance'] if lighting=='studio' or board_mode=='emissive' else scene['render']['environment_radiance']
    prop(env,'rgb','radiance',f'{radiance}, {radiance}, {radiance}')
    flat=[(obj.data,matrix) for obj,matrix in tree.walk()]
    camera_position=next(matrix[:3,3] for node,matrix in flat if node['type']=='camera')
    for obj, matrix in tree.walk():
        node = obj.data
        kind, ident = node['type'], node['id']
        if kind == 'spherical_shell':
            geometry = node['geometry']
            material = scene['materials'][node['material']]
            for suffix, radius, inverted in [('outer', geometry['outer_radius_m'], False),
                                            ('inner', geometry['outer_radius_m']-geometry['thickness_m'], True)]:
                shape = ET.SubElement(xml, 'shape', type='sphere', id=f'{ident}_{suffix}')
                prop(shape, 'float', 'radius', radius)
                prop(shape, 'boolean', 'flip_normals', inverted)
                xml_transform(shape, matrix)
                bsdf = ET.SubElement(shape, 'bsdf', type='dielectric')
                prop(bsdf, 'float', 'int_ior', material['ior'])
                prop(bsdf, 'float', 'ext_ior', material['outside_ior'])
        elif kind == 'checkerboard':
            g = node['geometry']
            material = scene['materials'][node['material']]
            row, col = np.indices((g['rows'], g['columns']))
            pixels = np.where((row + col) % 2 == 0, material['white'], material['black'])
            pixels = np.clip(pixels[::-1] * 255, 0, 255).round().astype('uint8')
            texture_name = f'{ident}_texture.png'
            Image.fromarray(np.repeat(pixels[:, :, None], 3, axis=2)).save(directory / texture_name)
            shape = ET.SubElement(xml, 'shape', type='rectangle', id=ident)
            scaled = matrix @ np.diag([g['columns']*g['cell_size_m']/2, g['rows']*g['cell_size_m']/2, 1, 1])
            xml_transform(shape, scaled)
            two_sided = ET.SubElement(shape, 'bsdf', type='twosided')
            diffuse = ET.SubElement(two_sided, 'bsdf', type='diffuse')
            if board_mode=='emissive':
                # Плоский area emitter односторонний. Нормаль направляем
                # к реальной камере, не меняя геометрию, UV и систему доски.
                if np.dot(matrix[:3,2],camera_position-matrix[:3,3])<0:
                    prop(shape,'boolean','flip_normals',True)
                prop(diffuse,'spectrum','reflectance',0.)
                emitter=ET.SubElement(shape,'emitter',type='area')
                texture=ET.SubElement(emitter,'texture',type='bitmap',name='radiance')
            else:
                texture=ET.SubElement(diffuse,'texture',type='bitmap',name='reflectance')
            prop(texture,'string','filename',texture_name)
            prop(texture,'string','filter_type','nearest')
            prop(texture,'boolean','raw',True)
        elif kind == 'camera':
            g = node['sensor']
            sensor = ET.SubElement(xml, 'sensor', type='perspective', id=ident)
            prop(sensor, 'float', 'fov', np.degrees(2*np.arctan(g['resolution'][0]/(2*g['focus_px']))))
            prop(sensor, 'string', 'fov_axis', 'x')
            prop(sensor, 'float', 'near_clip', g.get('near_clip_m', .00001))
            prop(sensor, 'float', 'far_clip', g.get('far_clip_m', 10000.))
            # Mitsuba: +Z forward, +Y up, +X left. Our camera: +X forward, +Z up.
            # This basis matches look_at(origin, origin+forward, up).
            sensor_matrix = matrix @ np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            xml_transform(sensor, sensor_matrix)
            film = ET.SubElement(sensor, 'film', type='hdrfilm')
            width, height = effective_resolution(scene, g)
            prop(film, 'integer', 'width', width)
            prop(film, 'integer', 'height', height)
            prop(film, 'string', 'pixel_format', 'rgb')
            prop(film, 'string', 'component_format', 'float32')
            sampler = ET.SubElement(sensor, 'sampler', type='independent')
            prop(sampler, 'integer', 'sample_count', scene['render']['samples'])
    if lighting=='studio' and board_mode=='diffuse':
        add_studio_light(xml,scene,flat)
    ET.indent(xml, space='  ')
    path = directory / 'scene.xml'
    ET.ElementTree(xml).write(path, encoding='utf-8', xml_declaration=True)
    return path



def render_image(mi, loaded, scene):
    settings=scene['render']
    samples=settings['samples']
    variant=settings.get('variant','scalar_rgb')
    if not variant.startswith('cuda'):
        return mi.render(loaded,spp=samples,seed=settings['seed']),{'passes':1,'execution_mode':'default'}
    import drjit as dr
    camera=next(obj.data for obj,_ in Scene.from_dict(scene).walk() if obj.data['type']=='camera')
    width,height=effective_resolution(scene,camera['sensor'])
    # В отличие от мегаядра wavefront обходит проблему OptiX в драйвере R535.
    # Ограничиваем число одновременно обрабатываемых путей на 12-ГБ GPU.
    budget=int(settings.get('gpu_max_rays_per_pass',4_000_000))
    if budget<1:
        raise ValueError('gpu_max_rays_per_pass должен быть положительным')
    per_pass=max(1,min(samples,budget//(width*height)))
    passes=(samples+per_pass-1)//per_pass
    result=np.zeros((height,width,3),dtype=np.float64)
    with dr.scoped_set_flag(dr.JitFlag.SymbolicLoops,False), dr.scoped_set_flag(dr.JitFlag.SymbolicCalls,False):
        done=0
        for index in range(passes):
            count=min(per_pass,samples-done)
            image=mi.render(loaded,spp=count,seed=(settings['seed']+index)%(2**32))
            # Копирование на CPU синхронизирует GPU, не оставляя очередь тяжёлых проходов.
            result+=np.asarray(image)*count
            done+=count
            if index==0 or index==passes-1 or (index+1)%max(1,passes//10)==0:
                print(f'GPU: сэмплы {done}/{samples}',flush=True)
    return (result/samples).astype(np.float32),{
        'passes':passes,'samples_per_pass':per_pass,'max_rays_per_pass':budget,
        'execution_mode':'wavefront','seed_per_pass':'(seed + pass_index) mod 2^32',
    }

def render_scene(scene, directory):
    path = export_scene(scene, directory)
    import mitsuba as mi
    variant = scene['render'].get('variant', 'scalar_rgb')
    mi.set_variant(variant)
    started = time.monotonic()
    image,execution=render_image(mi,mi.load_file(str(path.resolve())),scene)
    directory = Path(directory)
    scene=read_json(directory/'scene.json')
    mi.util.write_bitmap(str(directory / 'image.png'),image,write_async=False)
    camera = next(obj.data for obj, _ in Scene.from_dict(scene).walk() if obj.data['type']=='camera')
    write_json(directory / 'render_info.json', {
        'exporter_version': EXPORTER_VERSION, 'mitsuba_version': mi.__version__,
        'variant': variant, 'execution':execution, 'elapsed_seconds': time.monotonic()-started,
        'resolution': effective_resolution(scene, camera['sensor']),
        'samples': scene['render']['samples'], 'seed': scene['render']['seed'],
        'scene_hash': fingerprint(scene),
        'lighting_mode':scene['render']['lighting_mode'],
        'board_mode':scene['render']['board_mode'],
        'background_radiance':scene['render']['background_radiance'],
        'png_encoding':'8-bit RGB sRGB, clipped to [0,1]; full resolution',
    })


def append_manifest(directory, record):
    with (Path(directory) / 'manifest.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def render_batch(base, sweep, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # One writer per experiment; the OS releases the lock after interruption.
    import fcntl
    with (directory / '.render.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {'exporter_version': EXPORTER_VERSION, 'scene_hash': fingerprint(base), 'sweep_hash': fingerprint(sweep)}
        identity_path = directory / 'experiment.json'
        if identity_path.exists() and read_json(identity_path) != identity:
            raise ValueError('В этой папке другой эксперимент. Выберите новую папку.')
        write_json(identity_path, identity)
        write_json(directory / 'source_scene.json', base)
        write_json(directory / 'sweep.json', sweep)
        total = sweep_count(sweep, parameter_values(base))
        completed = {}
        manifest = directory / 'manifest.jsonl'
        if manifest.exists():
            for line in manifest.read_text().splitlines():
                try:
                    record = json.loads(line)
                    completed[record['index']] = record
                except (ValueError, KeyError):
                    continue  # an interrupted last line is safe to ignore
        failures = 0
        for index, values in enumerate(combinations(sweep, parameter_values(base)), 1):
            target = directory / f'{index:06d}'
            record = {'index': index, 'parameters': values, 'directory': target.name,
                      'time_utc': datetime.now(timezone.utc).isoformat()}
            previous = completed.get(index, {})
            if previous.get('status') == 'complete' and all((target/name).exists() for name in ['scene.json','scene.xml','image.png','render_info.json']) and ((target/'dome_position.json').exists() or (target/'ground_truth.json').exists()):
                if not (target/'calibmar_config.json').exists():
                    saved = read_json(target/'scene.json')
                    truth_path = target/'dome_position.json' if (target/'dome_position.json').exists() else target/'ground_truth.json'
                    write_json(target/'calibmar_config.json', calibmar_config(saved, read_json(truth_path)))
                print(f'{index}/{total}: уже готово', flush=True)
                continue
            print(f'{index}/{total}: {values}', flush=True)
            append_manifest(directory, {**record, 'status': 'running'})
            try:
                scene = resolve_scene(base, values)
                render_scene(scene, target)
                append_manifest(directory, {**record, 'status': 'complete', 'scene_hash': fingerprint(scene)})
            except Exception as exc:
                failures += 1
                append_manifest(directory, {**record, 'status': 'failed', 'error': str(exc)})
                print(f'Ошибка {index}: {exc}', flush=True)
                # A missing installation affects every frame; don't repeat it N times.
                if isinstance(exc, ImportError):
                    raise
        print(f'Завершено: {total} комбинаций, ошибок: {failures}', flush=True)
        if failures:
            raise RuntimeError(f'Не удалось отрендерить кадров: {failures}; см. manifest.jsonl')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sweep', type=Path)
    parser.add_argument('--step', type=Path, help='export full build123d assembly as STEP')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--variant',choices=['scalar_rgb','llvm_ad_rgb','cuda_ad_rgb'])
    parser.add_argument('--board-mode',choices=['diffuse','emissive'])
    parser.add_argument('--samples',type=int,help='samples per pixel')
    parser.add_argument('--width',type=int,help='render width in pixels')
    args = parser.parse_args()
    scene = read_json(args.scene)
    if args.variant:
        scene['render']['variant']=args.variant
    if args.board_mode:
        scene['render']['board_mode']=args.board_mode
    if args.samples is not None:
        if args.samples<1:
            parser.error('--samples должен быть положительным')
        scene['render']['samples']=args.samples
    if args.width is not None:
        if args.width<1:
            parser.error('--width должен быть положительным')
        camera=next(obj.data for obj,_ in Scene.from_dict(scene).walk() if obj.data['type']=='camera')
        scene['render']['resolution_scale']=args.width/camera['sensor']['resolution'][0]
    if args.sweep:
        if args.step:
            parser.error('--step применяется только к одной сцене')
        if args.export_only:
            parser.error('--export-only применяется только к одной сцене')
        render_batch(scene, read_json(args.sweep), args.out)
    else:
        # A saved resolved scene must not be recomputed: trust its explicit graph.
        if not scene.get('resolved') and 'parameterization' in scene:
            scene = resolve_scene(scene)
        if args.step:
            print(Scene.from_dict(scene).export_step(args.step))
        if args.export_only:
            print(export_scene(scene, args.out))
        else:
            render_scene(scene, args.out)
            print(f'Готово: {args.out}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Ошибка: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
