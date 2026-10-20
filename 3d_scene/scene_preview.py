"""OCP preview of the SAME resolved graph used by Mitsuba."""
from functools import lru_cache

import numpy as np
from build123d import (Align, Box, Color, Compound, Cone, Cylinder, Edge,
                      Plane, Pos, Sphere, Text)

from scene_cad import placement, shell_geometry


@lru_cache(maxsize=8)
def board_geometry(columns, rows, cell, white, black):
    tiles = []
    cell *= 1000
    for row in range(rows):
        for column in range(columns):
            tile = Pos((column-(columns-1)/2)*cell, (row-(rows-1)/2)*cell, 0) * Box(cell, cell, 0.05)
            value = white if (row+column)%2 == 0 else black
            tile.color = Color(value, value, value)
            tiles.append(tile)
    return Compound(label='Шахматная доска', children=tiles)


@lru_cache(maxsize=1)
def axes_geometry():
    arrows = []
    for label, direction, color in [('X',(1,0,0),'#df5055'), ('Y',(0,1,0),'#42a66c'), ('Z',(0,0,1),'#4285df')]:
        plane = Plane(origin=(0, 0, 0), z_dir=direction)
        shaft = Cylinder(6, 290, align=(Align.CENTER,Align.CENTER,Align.MIN))
        tip = Pos(0,0,290) * Cone(19,0,50,align=(Align.CENTER,Align.CENTER,Align.MIN))
        letter = Pos(*(np.array(direction,dtype=float)*390).tolist()) * Text(label,font_size=50)
        arrow = Compound(label=label, children=[plane.location*shaft,plane.location*tip,letter])
        arrow.color = Color(color)
        arrows.append(arrow)
    return Compound(label='Оси',children=arrows)


def camera_geometry(node, preview, compact=False):
    radius=preview['marker_radius_m']*1000
    marker=Sphere(radius*.5 if compact else radius)
    marker.color,marker.label=Color('#7099b2' if compact else '#ef3030'),'Оптический центр'
    # Одинаковый масштаб глубины и поперечных размеров сохраняет углы FOV.
    depth=preview['frustum_depth_m']*1000*(.5 if compact else 1)
    sensor=node['sensor']; w,h=sensor['resolution']; f=sensor['focus_px']
    corners=[(depth,sx*depth*w/(2*f),sy*depth*h/(2*f)) for sx,sy in [(-1,-1),(1,-1),(1,1),(-1,1)]]
    lines=[Edge.make_line((0,0,0),p) for p in corners]
    lines += [Edge.make_line(corners[i],corners[(i+1)%4]) for i in range(4)]
    wire=Compound(label='Поле зрения ×0.5' if compact else 'Поле зрения',children=lines)
    wire.color=Color('#7099b2' if compact else '#269caf')
    return Compound(label='Камера',children=[marker,wire])


@lru_cache(maxsize=32)
def polar_section_geometry(outer,thickness,angle_deg):
    """Вырез у полюса даёт устойчивую триангуляцию сферических граней OCCT."""
    if not np.isfinite(angle_deg) or not 1 <= angle_deg <= 170:
        raise ValueError('Размер сегмента: от 1 до 170 градусов')
    height=outer*1000*1.01
    radius=height*np.tan(np.radians(angle_deg/2))
    cutter=Cone(0,radius,height,align=(Align.CENTER,Align.CENTER,Align.MIN))
    return shell_geometry(outer,thickness) & cutter


@lru_cache(maxsize=64)
def section_geometry(outer,thickness,radial,angle_deg):
    # Сначала режем у +Z, затем переносим параметризацию поверхности вместе
    # с сегментом. Наклонять один конус до boolean недостаточно: OCCT может
    # выдать длинные треугольники с прогибом, сравнимым с толщиной стекла.
    direction=np.asarray(radial,dtype=float)
    direction/=np.linalg.norm(direction)
    axis=np.array([0.,0.,1.]) if abs(direction[2])<.99 else np.array([0.,1.,0.])
    x_dir=np.cross(axis,direction)
    orientation=Plane(origin=(0,0,0),x_dir=x_dir.tolist(),z_dir=direction.tolist()).location
    return polar_section_geometry(outer,thickness,angle_deg).moved(orientation)


def reference_geometry(preview):
    marker=Sphere(preview['marker_radius_m']*1500)
    marker.color=Color('#ff9f1c')
    lines=[]
    for axis,color in zip(np.eye(3),['#df5055','#42a66c','#4285df']):
        line=Edge.make_line((0,0,0),(axis*250).tolist())
        line.color=Color(color)
        lines.append(line)
    return Compound(label='Опорная камера',children=[marker,*lines])


def preview_shape(node,matrix,scene,reference_origin,compact=False,ghost=False):
    kind=node['type']; preview=scene['preview']
    if kind=='spherical_shell':
        g=node['geometry']
        shape=shell_geometry(g['outer_radius_m'],g['thickness_m'])
        if preview.get('shell_segment',preview['shell_mode']=='section'):
            # Один и тот же визуальный участок для всех совпадающих оболочек.
            local=(np.linalg.inv(matrix) @ np.append(reference_origin,1))[:3]
            radial=local/np.linalg.norm(local)
            shape=section_geometry(g['outer_radius_m'],g['thickness_m'],tuple(radial),preview.get('section_angle_deg',20.))
        shape=shape.moved(placement(matrix))
        shape.label='Оболочка'; shape.color=Color('#70b9e8')
        return shape
    if kind=='checkerboard':
        g=node['geometry']; material=scene['materials'][node['material']]
        if ghost:
            x,y=g['columns']*g['cell_size_m']*500,g['rows']*g['cell_size_m']*500
            corners=[(-x,-y,0),(x,-y,0),(x,y,0),(-x,y,0)]
            edges=[Edge.make_line(corners[i],corners[(i+1)%4]) for i in range(4)]
            edges += [Edge.make_line(corners[0],corners[2]),Edge.make_line(corners[1],corners[3])]
            shape=Compound(label='Контур шаблона',children=edges)
            shape.color=Color('#8c9299')
            return shape.moved(placement(matrix))
        return board_geometry(g['columns'],g['rows'],g['cell_size_m'],material['white'],material['black']).moved(placement(matrix))
    if kind=='camera':
        return camera_geometry(node,preview,compact).moved(placement(matrix))
    return reference_geometry(preview).moved(placement(matrix))


def reference_origin(scene):
    from scene_variants import visible_instances
    reference_id=scene.get('parameterization',{}).get('camera_reference_id','camera_reference')
    flat=list(visible_instances(scene))
    return next((m[:3,3] for n,m in flat if n['id']==reference_id),
                next(m[:3,3] for n,m in flat if n['type']=='camera'))


def build_preview(scene):
    from scene_variants import visible_instances
    flat=list(visible_instances(scene))
    origin=reference_origin(scene)
    objects=[preview_shape(n,m,scene,origin) for n,m in flat if n['type'] in ('spherical_shell','checkerboard')]
    objects += [preview_shape(n,m,scene,origin) for n,m in flat if n['type']=='camera']
    objects.append(axes_geometry())
    objects += [preview_shape(n,m,scene,origin) for n,m in flat if n['type']=='group']
    return objects


def build_multiple_preview(scene,sweep):
    from scene_variants import collect_preview_instances
    instances,shown,total=collect_preview_instances(scene,sweep,scene['preview'].get('variant_limit',100))
    buckets={'spherical_shell':[], 'checkerboard':[], 'camera':[], 'group':[]}
    origin=reference_origin(scene)
    for instance in instances:
        compact=not instance.current and not scene['preview'].get('variant_frustums',False)
        shape=preview_shape(instance.node,instance.matrix,scene,origin,compact,ghost=not instance.current)
        label='Текущая' if instance.current else 'Варианты '+','.join(map(str,instance.variants[:4]))
        if len(instance.variants)>4:
            label+='…'
        shape.label=f"{instance.node['id']} · {label}"
        buckets[instance.node['type']].append(shape)
    objects=[Compound(label=label,children=buckets[kind]) for kind,label in
             [('spherical_shell','Оболочки'),('checkerboard','Шаблоны'),('camera','Камеры')]]
    objects += [axes_geometry(),Compound(label='Опорные камеры',children=buckets['group'])]
    cameras=len(buckets['camera'])
    from scene_model import preview_axes,parameter_values
    from math import prod
    count=prod(map(len,preview_axes(sweep,parameter_values(scene),scene['preview'].get('full_display',False)).values()))
    message=f'Вариантов предпросмотра показано {shown:,} из {count:,}; полный перебор: {total:,}'
    message+=f'; уникальных камер: {cameras}. Текущая сцена включена отдельно.'
    return objects,message


def orbit_camera_options(scene,target,previous):
    options={'target':target.tolist()}
    previous=previous if isinstance(previous,dict) else {}
    position=np.asarray(previous.get('position',[]),dtype=float)
    old_target=np.asarray(previous.get('target',[]),dtype=float)
    if (position.shape==(3,) and old_target.shape==(3,)
            and np.isfinite(position).all() and np.isfinite(old_target).all()
            and np.linalg.norm(position-old_target)>1e-6):
        options['position']=(target+position-old_target).tolist()
        quaternion=np.asarray(previous.get('quaternion',[]),dtype=float)
        if quaternion.shape==(4,) and np.isfinite(quaternion).all():
            options['quaternion']=quaternion.tolist()
        zoom=previous.get('zoom')
        if isinstance(zoom,(int,float)) and np.isfinite(zoom) and zoom>0:
            options['zoom']=zoom
    else:
        # Первый запуск без предыдущего вида: явная изометрия вокруг камеры.
        distance=2.5*max(scene.get('parameters',{}).get('board_distance_m',2),
                         scene['preview'].get('section_radius_m',1.2),
                         scene['preview'].get('frustum_depth_m',1))*1000
        direction=np.array([1.,-1.,1.])/np.sqrt(3)
        options['position']=(target+direction*distance).tolist()
    return options


class Preview:
    def __init__(self):
        self.first=True
        self.last_target=None

    def show(self,scene,sweep=None):
        from ocp_vscode import Camera, Render, show, status
        mode=scene['preview']['shell_mode']
        if sweep is None:
            objects,message=build_preview(scene),'Просмотр обновлён.'
        else:
            objects,message=build_multiple_preview(scene,sweep)
        from scene_model import Scene
        camera_matrix=next(matrix for obj,matrix in Scene.from_dict(scene).walk()
                           if obj.data['type']=='camera')
        # Геометрия CAD и target Viewer используют миллиметры.
        target=camera_matrix[:3,3]*1000
        camera_options={}
        if self.last_target is None or not np.allclose(target,self.last_target,rtol=0,atol=1e-6):
            # Без position Viewer вызывает presetCamera('iso'), который заново
            # назначает центром вращения центр bounding box вместо target.
            try:
                previous=status()
            except Exception:
                previous={}
            camera_options=orbit_camera_options(scene,target,previous)
        shell_mode={'section':Render.FACES,'full':Render.FACES,'edges':Render.EDGES,'hidden':Render.NONE}[mode]
        show(*objects,names=['Оболочка','Шахматная доска','Камера','Оси','Опорная камера'],
             reset_camera=Camera.RESET if self.first else Camera.KEEP,
             modes=[shell_mode,Render.ALL,Render.ALL,Render.FACES,Render.ALL],
             axes=False,axes0=False,grid=False,
             clip_slider_0=1e9,clip_slider_1=1e9,clip_slider_2=1e9,
             clip_normal_0=[-1,0,0],clip_normal_1=[0,-1,0],clip_normal_2=[0,0,-1],
             clip_intersection=False,clip_planes=False,progress='',**camera_options)
        self.first=False
        self.last_target=target.copy()
        return message
