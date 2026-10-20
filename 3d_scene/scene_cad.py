"""Полное дерево JSON -> сборка build123d -> STEP (миллиметры)."""
from functools import lru_cache
from pathlib import Path

from build123d import Color, Compound, Edge, Face, Plane, Pos, Sphere, export_step

from scene_model import Scene


@lru_cache(maxsize=16)
def shell_geometry(outer, thickness):
    """Полная аналитическая оболочка; визуальный срез сюда не входит."""
    return Sphere(outer * 1000) - Sphere((outer - thickness) * 1000)


def placement(matrix):
    """Поза в метрах -> Location build123d в миллиметрах без смены осей."""
    return Plane(origin=(matrix[:3, 3] * 1000).tolist(),
                 x_dir=matrix[:3, 0].tolist(), z_dir=matrix[:3, 2].tolist()).location


def build_assembly(scene):
    """Сохраняет вложенные группы и локальные позы, включая позу корня.

    Доска экспортируется как набор плоских граней без выдуманной толщины.
    Камера — каркас пирамиды, а не материальный корпус. Материалы оптики
    остаются в JSON: STEP передаёт геометрию, имена, цвета и иерархию.
    """
    tree = scene if isinstance(scene, Scene) else Scene.from_dict(scene)
    materials = tree.data['materials']

    def build(obj):
        node = obj.data
        parts = []
        kind = node['type']
        if kind == 'spherical_shell':
            g = node['geometry']
            shell = shell_geometry(g['outer_radius_m'], g['thickness_m']).moved(placement_identity)
            shell.label = 'geometry'
            shell.color = Color('#70b9e8')
            parts.append(shell)
        elif kind == 'checkerboard':
            g = node['geometry']
            material = materials[node['material']]
            cell = g['cell_size_m'] * 1000
            for row in range(g['rows']):
                for column in range(g['columns']):
                    face = Pos((column - (g['columns'] - 1) / 2) * cell,
                               (row - (g['rows'] - 1) / 2) * cell, 0) * Face.make_rect(cell, cell)
                    face.label = f'cell_{row}_{column}'
                    value = material['white'] if (row + column) % 2 == 0 else material['black']
                    face.color = Color(value, value, value)
                    parts.append(face)
        elif kind == 'camera':
            sensor = node['sensor']
            depth = tree.data.get('preview', {}).get('frustum_depth_m', 1.) * 1000
            w, h = sensor['resolution']
            f = sensor['focus_px']
            corners = [(depth, sx * depth * w / (2*f), sy * depth * h / (2*f))
                       for sx, sy in [(-1,-1), (1,-1), (1,1), (-1,1)]]
            parts = [Edge.make_line((0,0,0), corner) for corner in corners]
            parts += [Edge.make_line(corners[i], corners[(i+1) % 4]) for i in range(4)]
        elif kind == 'group' and node['id'] == tree.data.get('parameterization', {}).get('camera_reference_id', 'camera_reference'):
            parts = [Edge.make_line((0,0,0), end) for end in [(250,0,0),(0,250,0),(0,0,250)]]
        elif kind != 'group':
            raise ValueError(f'Неизвестный тип узла: {kind}')
        parts.extend(build(child) for child in obj.children)
        return Compound(label=node['id'], children=parts).moved(placement(obj.local_matrix()))

    # moved создаёт независимую копию кэшированной оболочки для каждой сборки.
    from build123d import Location
    placement_identity = Location()
    return build(tree.root)


def export_step_scene(scene, path):
    """STEP + соседний .scene.json с точными параметрами и материалами."""
    tree = scene if isinstance(scene, Scene) else Scene.from_dict(scene)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not export_step(build_assembly(tree), path):
        raise RuntimeError(f'Не удалось записать STEP: {path}')
    tree.save(path.with_suffix('.scene.json'))
    return path
