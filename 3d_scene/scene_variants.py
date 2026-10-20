"""Выборка вариантов и объединение совпадающих мировых экземпляров без CAD."""
from dataclasses import dataclass, field
import numpy as np

from scene_model import Scene, fingerprint, parameter_values, resolve_scene, sampled_combinations


@dataclass
class PreviewInstance:
    node: dict
    matrix: np.ndarray
    current: bool = False
    variants: list[int] = field(default_factory=list)


def visible_instances(scene):
    reference_id = scene.get('parameterization', {}).get('camera_reference_id', 'camera_reference')
    for obj, matrix in Scene.from_dict(scene).walk():
        node = obj.data
        if node['type'] != 'group' or node['id'] == reference_id:
            yield node, matrix


def instance_key(scene, node, matrix):
    # IOR не имеет видимого действия в CAD. Материалы доски имеют.
    material = scene.get('materials', {}).get(node.get('material'), {})
    appearance = {key:material[key] for key in ['white','black'] if key in material}
    rounded=np.round(matrix,8)
    rounded[np.abs(rounded)<1e-8]=0
    return fingerprint({
        'id':node['id'], 'type':node['type'],
        'matrix':rounded.tolist(),
        'geometry':node.get('geometry'), 'sensor':node.get('sensor'),
        'appearance':appearance,
    })


def collect_preview_instances(scene, sweep, limit=100):
    values, total = sampled_combinations(sweep, parameter_values(scene), limit,
                                         default_full=scene['preview'].get('full_display',False))
    unique = {}
    for node,matrix in visible_instances(scene):
        unique[instance_key(scene,node,matrix)] = PreviewInstance(node,matrix,current=True)
    for index,parameters in enumerate(values,1):
        variant = resolve_scene(scene,parameters)
        for node,matrix in visible_instances(variant):
            key = instance_key(variant,node,matrix)
            if key not in unique:
                unique[key] = PreviewInstance(node,matrix)
            unique[key].variants.append(index)
    return list(unique.values()), len(values), total
