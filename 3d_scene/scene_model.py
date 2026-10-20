"""Readable scene graph, SI units, explicit transforms and reproducible sweeps."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# -----------------------------------------------------------------------------
# Общие преобразования координат и поворотов
# Входные и выходные углы — в градусах; радианы используются внутри вычислений.
# -----------------------------------------------------------------------------


def euler_deg_2_mat33(angles):
    """Углы [X, Y, Z] в градусах → матрица поворота 3×3."""
    # Строчные 'xyz': повороты вокруг неподвижных осей X, затем Y, затем Z.
    # Это прежний порядок OpenSCAD: Rz @ Ry @ Rx для векторов-столбцов.
    return Rotation.from_euler('xyz', angles, degrees=True).as_matrix()


def mat33_2_euler_deg(r):
    """Матрица поворота 3×3 → углы [X, Y, Z] в градусах для JSON."""
    # Используем тот же порядок осей, что и в euler_deg_2_mat33(). При Y = ±90°
    # углы неоднозначны: SciPy выбирает Z = 0, сохраняя сам поворот.
    return Rotation.from_matrix(r).as_euler('xyz', degrees=True).tolist()


def sph_deg_2_xyz(radius, pitch_deg, yaw_deg):
    """Возвращает XYZ: pitch — угол над XY, yaw — азимут от +X к +Y.

    Углы заданы в градусах; XYZ имеют те же единицы, что и radius.
    """
    pitch_rad, yaw_rad = math.radians(pitch_deg), math.radians(yaw_deg)
    return np.array([radius * math.cos(pitch_rad) * math.cos(yaw_rad),
                     radius * math.cos(pitch_rad) * math.sin(yaw_rad), radius * math.sin(pitch_rad)])


# -----------------------------------------------------------------------------
# Данные сцены, JSON и преобразования узлов
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True, kw_only=True)
class ParameterSpec:
    """Описание параметра для интерфейса, отдельно от его текущего значения.

    display_factor переводит значение сцены в единицы, показанные в панели.
    Поля задаются по имени, чтобы не путать подпись, единицу и множитель.
    """

    label: str
    unit: str
    display_factor: float = 1.0


PARAMETERS: dict[str, ParameterSpec] = {
    'pitch_deg': ParameterSpec(label='Тангаж', unit='°'),
    'yaw_deg': ParameterSpec(label='Рыскание', unit='°'),
    'roll_deg': ParameterSpec(label='Крен', unit='°'),
    'board_distance_m': ParameterSpec(label='Расстояние до доски', unit='см', display_factor=100),
    'thickness_m': ParameterSpec(label='Толщина стекла', unit='см', display_factor=100),
    'ior': ParameterSpec(label='Показатель преломления', unit=''),
}


OFFSET_KEYS = {
    'camera_dx_m': ('translation_m', 0),
    'camera_dy_m': ('translation_m', 1),
    'camera_dz_m': ('translation_m', 2),
    'camera_rx_deg': ('rotation_deg', 0),
    'camera_ry_deg': ('rotation_deg', 1),
    'camera_rz_deg': ('rotation_deg', 2),
}
for key, label in zip(OFFSET_KEYS, ['ΔX', 'ΔY', 'ΔZ', 'Крен', 'Тангаж', 'Рыскание']):
    PARAMETERS[key] = ParameterSpec(label=label, unit='см' if key.endswith('_m') else '°',
                                    display_factor=100 if key.endswith('_m') else 1)


def parameter_values(scene):
    """Все текущие параметры, включая люфт старых документов без новых ключей."""
    result = dict(scene['parameters'])
    offsets = scene['parameterization'].get('camera_offset', {})
    for key, (vector, index) in OFFSET_KEYS.items():
        result.setdefault(key, offsets.get(vector, [0, 0, 0])[index])
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


class Object3D:
    """Узел JSON: локальная поза, произвольные данные и дочерние объекты.

    Группы и примитивы используют одну композицию преобразований. Объект
    ссылается на исходный словарь: правки параметров видны без второго графа.
    """

    def __init__(self, data, parent=None):
        self.data = data
        self.parent = parent
        self.children = [Object3D(child, self) for child in data.get('children', [])]

    @classmethod
    def from_dict(cls, data):
        return cls(deepcopy(data))

    def to_dict(self):
        data = deepcopy(self.data)
        if self.children or 'children' in data:
            data['children'] = [child.to_dict() for child in self.children]
        return data

    def local_matrix(self):
        """T @ Rz @ Ry @ Rx, длины в метрах, углы в градусах."""
        transform = self.data.get('transform', {})
        rotation = np.asarray(transform.get('rotation_deg', [0, 0, 0]), dtype=float)
        translation = np.asarray(transform.get('translation_m', [0, 0, 0]), dtype=float)
        if (rotation.shape != (3,) or translation.shape != (3,)
                or not np.isfinite(rotation).all() or not np.isfinite(translation).all()):
            raise ValueError('Поза узла: нужны два вектора из трёх конечных чисел')
        matrix = np.eye(4)
        matrix[:3, :3] = euler_deg_2_mat33(rotation)
        matrix[:3, 3] = translation
        return matrix

    def world_matrix(self):
        local = self.local_matrix()
        return local if self.parent is None else self.parent.world_matrix() @ local

    def walk(self, parent_matrix=None):
        """Обход поддерева: (объект, мировая матрица), включая себя."""
        if parent_matrix is None:
            parent_matrix = np.eye(4) if self.parent is None else self.parent.world_matrix()
        world = parent_matrix @ self.local_matrix()
        yield self, world
        for child in self.children:
            yield from child.walk(world)


# Соглашения версии 1. Старые JSON без поля conventions используют эти же правила.
CONVENTIONS = {
    'handedness': 'right', 'vectors': 'column',
    'local_matrix': 'T @ Rz @ Ry @ Rx', 'world_matrix': 'parent @ local',
    'camera_forward': '+X', 'camera_up': '+Z',
    'board_plane': 'XY', 'board_normal': '+Z', 'board_origin': 'center',
    'board_white_cell': '(row + column) % 2 == 0',
}


class Scene:
    """Самодостаточный документ и объектное дерево, общее для экспортов."""

    def __init__(self, document):
        self.data = document
        self.root = Object3D(document['graph'])

    @classmethod
    def from_dict(cls, document):
        validate_scene(document)
        return cls(deepcopy(document))

    @classmethod
    def load(cls, path):
        return cls.from_dict(read_json(path))

    def walk(self):
        return self.root.walk()

    def to_dict(self):
        data = deepcopy(self.data)
        data['graph'] = self.root.to_dict()
        data['conventions'] = deepcopy(CONVENTIONS)
        # Явные параметры камеры, ранее фиксированные в XML-экспортере.
        for obj, _ in Object3D(data['graph']).walk():
            if obj.data['type'] == 'camera':
                sensor = obj.data['sensor']
                sensor.setdefault('near_clip_m', 0.00001)
                sensor.setdefault('far_clip_m', 10000.0)
        return data

    def save(self, path):
        write_json(path, self.to_dict())

    def to_build123d(self):
        from scene_cad import build_assembly
        return build_assembly(self)

    def export_step(self, path):
        from scene_cad import export_step_scene
        return export_step_scene(self, path)


def nodes_by_id(scene):
    nodes = [obj.data for obj, _ in Object3D(scene['graph']).walk()]
    ids = [node['id'] for node in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError('Идентификаторы узлов должны быть уникальны')
    return dict(zip(ids, nodes))


def resolve_scene(base, values=None):
    """Evaluate parameter dependencies; saved graph is sufficient for both backends."""
    scene = deepcopy(base)
    params = parameter_values(scene)
    scene['parameters'] = params
    if values:
        if set(values) - set(PARAMETERS):
            raise ValueError('Неизвестные параметры сцены')
        params.update(values)
    if set(params) != set(PARAMETERS) or any(not math.isfinite(float(v)) for v in params.values()):
        raise ValueError('Нужны конечные числовые параметры сцены')
    nodes = nodes_by_id(scene)
    binding = scene['parameterization']
    shell = nodes[binding['shell_id']]
    camera = nodes[binding['camera_id']]
    reference = nodes[binding['board_reference_id']]
    board = nodes[binding['board_id']]
    outer = shell['geometry']['outer_radius_m']
    thickness, gap = params['thickness_m'], binding['camera_clearance_m']
    if not 0 < thickness < outer or gap < 0 or outer-thickness-gap <= 0.01:
        raise ValueError('Толщина/зазор не оставляют места камере внутри сферы')
    if params['ior'] < 1 or params['board_distance_m'] <= 0:
        raise ValueError('Показатель преломления должен быть ≥1, расстояние >0')
    if not -89.9 <= params['pitch_deg'] <= 89.9:
        raise ValueError('Тангаж должен быть в пределах −89.9…89.9°')
    shell['geometry']['thickness_m'] = thickness
    scene['materials'][shell['material']]['ior'] = params['ior']
    radius = outer - thickness - gap
    origin = sph_deg_2_xyz(radius, params['pitch_deg'], params['yaw_deg'])
    anchor = origin.copy()
    reference['transform'] = {'translation_m': anchor.tolist(), 'rotation_deg': [0, 0, 0]}
    board['transform']['translation_m'] = [params['board_distance_m'], 0, 0]
    camera['transform']['translation_m'] = origin.tolist()
    mode = binding['camera_aim']
    if mode == 'radial':
        angles = [params['roll_deg'], -params['pitch_deg'], params['yaw_deg']]
    elif mode == 'fixed_world':
        angles = [params['roll_deg'], 0, params['yaw_deg']]
    elif mode == 'board_center':
        forward = euler_deg_2_mat33([0, 0, params['yaw_deg']])[:, 0]
        forward /= np.linalg.norm(forward)
        right = np.cross([0, 0, 1], forward)
        if np.linalg.norm(right) < 1e-8:
            raise ValueError('Оптическая ось параллельна вертикали')
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        angles = mat33_2_euler_deg(np.column_stack([forward, right, up]) @ euler_deg_2_mat33([params['roll_deg'], 0, 0]))
    else:
        raise ValueError(f'Неизвестное правило направления камеры: {mode}')
    # Номинальная поза задаёт установку шаблона; люфт действует только на камеру.
    nominal_rotation = euler_deg_2_mat33(angles)
    reference['transform']['rotation_deg'] = angles
    rig = next(obj.data for obj, _ in Object3D(scene['graph']).walk()
               if any(child.data['id'] == camera['id'] for child in obj.children))
    reference_id = binding.get('camera_reference_id', 'camera_reference')
    binding['camera_reference_id'] = reference_id
    reference_node = nodes.get(reference_id)
    if reference_node is None:
        reference_node = {'id': reference_id, 'type': 'group'}
        rig.setdefault('children', []).append(reference_node)
    reference_node['transform'] = {'translation_m': origin.tolist(), 'rotation_deg': angles}
    offset = {'translation_m': [0., 0., 0.], 'rotation_deg': [0., 0., 0.]}
    for key, (vector, index) in OFFSET_KEYS.items():
        offset[vector][index] = params[key]
    binding['camera_offset'] = offset
    translation = np.asarray(offset.get('translation_m', [0, 0, 0]), dtype=float)
    rotation = np.asarray(offset.get('rotation_deg', [0, 0, 0]), dtype=float)
    if translation.shape != (3,) or rotation.shape != (3,) or not np.isfinite(translation).all() or not np.isfinite(rotation).all():
        raise ValueError('Люфт камеры: нужны два вектора из трёх конечных чисел')
    camera['transform']['translation_m'] = (origin + nominal_rotation @ translation).tolist()
    camera['transform']['rotation_deg'] = mat33_2_euler_deg(nominal_rotation @ euler_deg_2_mat33(rotation))
    scene['resolved'] = True
    return Scene.from_dict(scene).to_dict()


def validate_scene(scene):
    if scene.get('schema_version') != 1 or scene.get('units') != {'length': 'm', 'angle': 'deg'}:
        raise ValueError('Ожидается schema_version=1, метры и градусы')
    nodes_by_id(scene)
    counts = {'camera': 0, 'spherical_shell': 0, 'checkerboard': 0}
    if scene.get('conventions', CONVENTIONS) != CONVENTIONS:
        raise ValueError('Неподдерживаемые соглашения координат')
    for obj, matrix in Object3D(scene['graph']).walk():
        node = obj.data
        if node['type'] not in ('group', *counts):
            raise ValueError(f"Неизвестный тип узла: {node['type']}")
        if not np.isfinite(matrix).all():
            raise ValueError('Неконечное преобразование')
        if node['type'] in counts:
            counts[node['type']] += 1
        if node['type'] == 'spherical_shell':
            g = node['geometry']
            if not 0 < g['thickness_m'] < g['outer_radius_m']:
                raise ValueError('Неверные радиусы оболочки')
        if node['type'] == 'checkerboard':
            g = node['geometry']
            if any(not isinstance(g[k], int) or g[k] < 1 for k in ['columns', 'rows']) or g['cell_size_m'] <= 0:
                raise ValueError('Неверный размер доски')
        if node['type'] == 'camera':
            g = node['sensor']
            near, far = g.get('near_clip_m', .00001), g.get('far_clip_m', 10000.)
            if not math.isfinite(far) or not 0 < near < far:
                raise ValueError('Неверные плоскости отсечения камеры')
            if len(g['resolution']) != 2 or any(not isinstance(v, int) or v < 1 for v in g['resolution']) or g['focus_px'] <= 0:
                raise ValueError('Неверные параметры камеры')
    if counts['camera'] != 1:
        raise ValueError('Нужна ровно одна камера')
    render = scene['render']
    if any(not isinstance(render[k], int) or render[k] < 1 for k in ['samples', 'max_depth']):
        raise ValueError('Число сэмплов и глубина должны быть положительными целыми')
    if not isinstance(render['seed'], int) or render['seed'] < 0:
        raise ValueError('Seed должен быть неотрицательным целым')
    return scene


def range_values(spec):
    start, stop, step = [Decimal(str(spec[k])) for k in ['min', 'max', 'step']]
    if not all(v.is_finite() for v in [start, stop, step]) or step <= 0 or stop < start:
        raise ValueError('Диапазон: min ≤ max, шаг > 0')
    count = int((stop - start) // step) + 1
    if count > 100000:
        raise ValueError('Слишком много значений одного параметра (>100000)')
    return [float(start + i * step) for i in range(count)]


def sweep_axes(sweep, current=None):
    ranges = sweep.get('ranges', {})
    if (sweep.get('schema_version') != 1 or set(ranges) - set(PARAMETERS)
            or (set(PARAMETERS) - set(OFFSET_KEYS)) - set(ranges)):
        raise ValueError('Неверные параметры перебора')
    axes = {}
    for key in PARAMETERS:
        spec = ranges.get(key)
        if spec is None:
            axes[key] = [float((current or {}).get(key, 0))]
        elif spec.get('enabled', True):
            axes[key] = range_values(spec)
        else:
            value = (current or {}).get(key, spec.get('value'))
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f'Нужно текущее значение выключенного параметра {key}')
            axes[key] = [float(value)]
    return axes


def sweep_count(sweep, current=None):
    return math.prod(len(values) for values in sweep_axes(sweep, current).values())


def combination_at(axes, index):
    """Индекс в декартовом произведении без создания всего произведения."""
    row = {}
    for key, values in reversed(list(axes.items())):
        index, digit = divmod(index, len(values))
        row[key] = values[digit]
    return {key: row[key] for key in axes}


def preview_axes(sweep, current, default_full=False):
    axes=sweep_axes(sweep,current)
    for key,values in axes.items():
        if not sweep['ranges'].get(key,{}).get('preview_full',default_full) and len(values)>1:
            axes[key]=[values[0],values[-1]]
    return axes


def sampled_combinations(sweep, current, limit, default_full=False):
    if not isinstance(limit, int) or not 1 <= limit <= 500:
        raise ValueError('Лимит предпросмотра: от 1 до 500 вариантов')
    total=sweep_count(sweep,current)
    axes=preview_axes(sweep,current,default_full)
    candidate_count=math.prod(map(len,axes.values()))
    count=min(limit,candidate_count)
    indices=[0] if count==1 else [i*(candidate_count-1)//(count-1) for i in range(count)]
    return [combination_at(axes,index) for index in indices],total


def combinations(sweep, current=None):
    axes = sweep_axes(sweep, current)
    for row in itertools.product(*axes.values()):
        yield dict(zip(axes, row))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
