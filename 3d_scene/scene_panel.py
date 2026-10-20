"""Editable parameter table, sweep selection, overlay preview and resumable rendering."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

from scene_model import (ROOT, read_json, resolve_scene,
                         sweep_axes, sweep_count, write_json, parameter_values)

AIMS = {'Вдоль +X, рыскание вокруг Z': 'fixed_world',
        'Вдоль радиуса наружу': 'radial',
        'На центр доски': 'board_center'}
BOARD_MODES={'Обычная':'diffuse','Светящаяся':'emissive'}
BACKENDS={'CPU':'scalar_rgb','CPU (LLVM)':'llvm_ad_rgb','GPU NVIDIA (CUDA)':'cuda_ad_rgb'}
MODES = {'Поверхность': 'full', 'Рёбра': 'edges', 'Скрыть': 'hidden'}


class ScenePanel:
    def __init__(self, main_window):
        self.main_window = main_window
        self.base = read_json(ROOT/'scene.json')
        self.sweep = read_json(ROOT/'sweep.json')
        self.state_path=ROOT/'panel_state.json'
        self.autosave_job=None
        self.state_dirty=False
        self.loading_state=True
        self.restore_error=None
        if self.state_path.exists():
            try:
                state=read_json(self.state_path)
                if state.get('schema_version')!=1:
                    raise ValueError('Неизвестная версия состояния панели')
                scene,sweep=state['scene'],state['sweep']
                resolve_scene(scene)
                sweep_axes(sweep,parameter_values(scene))
                self.base,self.sweep=scene,sweep
            except (OSError,ValueError,KeyError,TypeError,AttributeError) as exc:
                self.restore_error=f'Не удалось восстановить panel_state.json: {exc}'
        self.axes = sweep_axes(self.sweep,parameter_values(self.base))
        self.values = parameter_values(self.base)
        # Просмотр обновляется в одном фоновом потоке, чтобы не блокировать окно.
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.preview = None
        self.preview_future = None
        self.pending_scene = None
        self.pending_at = 0
        self.job = None
        self.job_stopped = False
        # Поток чтения рендера передаёт сюда строки; виджеты меняет главный поток.
        self.job_queue = queue.Queue()
        self.closing = False
        self.last_directory = None
        # Загрузка полей не должна запускать частичные обновления сцены.
        self.suppress = False
        self.main_window.title('Параметрическая сцена — JSON / OCP / Mitsuba')
        self.main_window.geometry('1000x850')
        self.main_window.minsize(920,760)
        # Первый столбец получает свободное место при увеличении ширины окна.
        self.main_window.columnconfigure(0,weight=1)
        self.status = tk.StringVar(value='Загрузка…')
        self.preview_status = tk.StringVar()
        self.count = tk.StringVar()
        self.pinned = tk.BooleanVar(value=True)
        self.main_window.attributes('-topmost', True)
        self.ui_settings_path=ROOT/'ui_settings.json'
        try:
            settings=read_json(self.ui_settings_path)
            size=settings.get('font_size',10)
            if type(size) is not int or not 8 <= size <= 24:
                size=10
        except (OSError,ValueError,AttributeError):
            size=10
        self.font_size=tk.StringVar(value=str(size))
        self.applied_font_size=size
        self.set_font_size(size)
        self.build_ui()
        self.table.resize_columns()
        self.load_controls()
        self.loading_state=False
        self.pinned.set(self.base.get('panel',{}).get('pinned',True))
        self.main_window.attributes('-topmost',self.pinned.get())
        for variable in (self.samples,self.width,self.preview_limit,self.pinned,self.section_angle):
            variable.trace_add('write',lambda *_:self.queue_autosave())
        self.status.set(self.restore_error or 'Рабочие настройки сохраняются автоматически.')
        # При закрытии окна сначала останавливаем фоновые задачи.
        self.main_window.protocol('WM_DELETE_WINDOW', self.close)
        self.poll_job=self.main_window.after(100, self.poll)

    def build_ui(self):
        frame = ttk.Frame(self.main_window,padding=12)
        frame.grid(row=1,column=0,sticky='nsew')
        frame.columnconfigure(0,weight=1)
        from parameter_table import ParameterTable
        self.table = ParameterTable(frame, self.table_changed, lambda exc: self.preview_status.set(f'Ввод: {exc}'))
        self.table.grid(row=0,column=0,sticky='nsew')
        frame.rowconfigure(0,weight=1)
        self.main_window.rowconfigure(1,weight=1)
        actions=ttk.Frame(frame)
        actions.grid(row=1,column=0,sticky='ew',pady=8)
        ttk.Button(actions,text='Применить диапазоны',command=self.apply_ranges).pack(side='left',padx=(0,8))
        ttk.Button(actions,text='Сохранить настройки',command=self.save_settings).pack(side='left',padx=4)
        ttk.Button(actions,text='Перечитать JSON',command=self.reload).pack(side='left',padx=4)
        ttk.Label(actions,textvariable=self.count).pack(side='right')
        view=ttk.LabelFrame(frame,text='Просмотр (не меняет физическую геометрию экспорта)',padding=10)
        view.grid(row=2,column=0,sticky='ew',pady=4)
        ttk.Label(view,text='Оболочка:').grid(row=0,column=0,sticky='w')
        self.mode=tk.StringVar()
        mode_box=ttk.Combobox(view,textvariable=self.mode,values=list(MODES),state='readonly',width=18)
        mode_box.grid(row=0,column=1,padx=8)
        mode_box.bind('<<ComboboxSelected>>',lambda _:self.queue_preview())
        ttk.Label(view,text='Направление опорной камеры:').grid(row=0,column=2,padx=(14,0))
        self.aim=tk.StringVar()
        aim_box=ttk.Combobox(view,textvariable=self.aim,values=list(AIMS),state='readonly',width=32)
        aim_box.grid(row=0,column=3,padx=8)
        aim_box.bind('<<ComboboxSelected>>',lambda _:self.queue_preview())
        ttk.Checkbutton(view,text='Поверх окон',variable=self.pinned,command=lambda:self.main_window.attributes('-topmost',self.pinned.get())).grid(row=1,column=0,columnspan=2,sticky='w',pady=(10,0))
        ttk.Label(view,text='IOR виден на рендере; в OCP меняется только геометрия.').grid(row=1,column=2,columnspan=2,pady=(10,0),sticky='w')
        self.multiple=tk.BooleanVar(value=False)
        self.full_frustums=tk.BooleanVar(value=False)
        self.preview_limit=tk.StringVar(value='100')
        ttk.Checkbutton(view,text='Показать варианты перебора',variable=self.multiple,command=self.queue_preview).grid(row=2,column=0,columnspan=2,sticky='w',pady=6)
        ttk.Checkbutton(view,text='Пирамиды вариантов ×1',variable=self.full_frustums,command=self.queue_preview).grid(row=2,column=2,sticky='w')
        limit=ttk.Frame(view)
        limit.grid(row=2,column=3,sticky='w')
        ttk.Label(limit,text='Лимит:').pack(side='left')
        limit_entry=ttk.Spinbox(limit,from_=1,to=500,textvariable=self.preview_limit,width=6,command=self.queue_preview)
        limit_entry.pack(side='left',padx=5)
        limit_entry.bind('<Return>',lambda _:self.queue_preview())
        self.shell_segment=tk.BooleanVar(value=True)
        self.section_angle=tk.StringVar(value='20')
        ttk.Checkbutton(view,text='Сегмент оболочки',variable=self.shell_segment,command=self.queue_preview).grid(row=3,column=0,columnspan=2,sticky='w',pady=4)
        segment=ttk.Frame(view)
        segment.grid(row=3,column=2,columnspan=2,sticky='w')
        ttk.Label(segment,text='Угловой диаметр, °:').pack(side='left')
        angle_entry=ttk.Spinbox(segment,from_=1,to=170,increment=1,textvariable=self.section_angle,width=6,command=self.queue_preview)
        angle_entry.pack(side='left',padx=5)
        angle_entry.bind('<Return>',lambda _:self.queue_preview())
        angle_entry.bind('<FocusOut>',lambda _:self.queue_preview())
        render=ttk.LabelFrame(frame,text='Mitsuba',padding=10)
        render.grid(row=3,column=0,sticky='ew',pady=10)
        self.samples=tk.StringVar()
        self.width=tk.StringVar()
        ttk.Label(render,text='Сэмплов/пиксель:').grid(row=0,column=0)
        ttk.Entry(render,textvariable=self.samples,width=8).grid(row=0,column=1,padx=8)
        ttk.Label(render,text='Ширина рендера, px:').grid(row=0,column=2,padx=(16,0))
        ttk.Entry(render,textvariable=self.width,width=8).grid(row=0,column=3,padx=8)
        ttk.Label(render,text='Вычисления:').grid(row=1,column=0,sticky='w',pady=(6,0))
        self.backend=tk.StringVar(value='CPU')
        backend_box=ttk.Combobox(render,textvariable=self.backend,values=list(BACKENDS),state='readonly',width=21)
        backend_box.grid(row=1,column=1,sticky='w',pady=(6,0))
        backend_box.bind('<<ComboboxSelected>>',lambda _:self.queue_autosave())
        ttk.Label(render,text='Доска:').grid(row=1,column=2,sticky='w',padx=(16,0),pady=(6,0))
        self.board_mode=tk.StringVar(value='Обычная')
        board_box=ttk.Combobox(render,textvariable=self.board_mode,values=list(BOARD_MODES),state='readonly',width=15)
        board_box.grid(row=1,column=3,sticky='w',pady=(6,0))
        board_box.bind('<<ComboboxSelected>>',lambda _:self.queue_autosave())
        buttons=ttk.Frame(render)
        buttons.grid(row=2,column=0,columnspan=5,pady=(12,0),sticky='w')
        ttk.Button(buttons,text='Экспорт JSON / XML',command=self.export_current).pack(side='left',padx=(0,6))
        ttk.Button(buttons,text='Экспорт STEP',command=self.export_cad).pack(side='left',padx=6)
        ttk.Button(buttons,text='Рендерить текущую',command=lambda:self.start_job(False)).pack(side='left',padx=6)
        ttk.Button(buttons,text='Рендерить набор',command=lambda:self.start_job(True)).pack(side='left',padx=6)
        ttk.Button(buttons,text='Продолжить…',command=self.resume).pack(side='left',padx=6)
        ttk.Button(buttons,text='Стоп',command=self.stop_job).pack(side='left',padx=6)
        ttk.Label(frame,textvariable=self.preview_status,wraplength=920).grid(row=4,column=0,sticky='w',pady=5)
        ttk.Label(frame,textvariable=self.status,wraplength=920).grid(row=5,column=0,sticky='w',pady=5)
        self.log=tk.Text(frame,height=3,width=100,state='disabled')
        self.log.grid(row=6,column=0,sticky='ew',pady=4)
        preferences=ttk.Frame(self.main_window,padding=(12,8))
        preferences.grid(row=0,column=0,sticky='ew')
        ttk.Label(preferences,text='Размер шрифта, пт:').pack(side='left')
        font_entry=ttk.Spinbox(preferences,from_=8,to=24,textvariable=self.font_size,
                             width=4,command=self.apply_font_size)
        self.font_entry=font_entry
        font_entry.pack(side='left',padx=5)
        font_entry.bind('<Return>',lambda _:self.apply_font_size())
        font_entry.bind('<FocusOut>',lambda _:self.apply_font_size())
        ttk.Button(preferences,text='Применить',command=self.apply_font_size).pack(side='left')
        ttk.Label(preferences,text='Сохраняется автоматически').pack(side='left',padx=10)

    def set_font_size(self,size):
        # Именованные шрифты обновляют уже созданные виджеты автоматически.
        for name in tkfont.names(self.main_window):
            if name.startswith('Tk'):
                tkfont.nametofont(name,root=self.main_window).configure(size=size)
        style=ttk.Style(self.main_window)
        style.configure('.',font='TkDefaultFont')
        style.configure('Treeview',font='TkDefaultFont',
                        rowheight=tkfont.nametofont('TkDefaultFont',root=self.main_window).metrics('linespace')+8)
        style.configure('Treeview.Heading',font='TkHeadingFont')

    def apply_font_size(self):
        try:
            size=int(self.font_size.get())
            if not 8 <= size <= 24:
                raise ValueError('Размер шрифта: от 8 до 24 пунктов')
            if size==self.applied_font_size:
                return
            write_json(self.ui_settings_path,{'font_size':size})
            self.set_font_size(size)
            self.applied_font_size=size
            self.table.resize_columns()
            self.status.set(f'Размер шрифта {size} пт сохранён.')
        except (ValueError,OSError) as exc:
            self.font_size.set(str(self.applied_font_size))
            self.status.set(str(exc))

    def load_controls(self):
        # Заполнение полей тоже вызывает события; не обновляем сцену частями.
        self.suppress=True
        try:
            self.table.load(self.base,self.sweep)
            self.values = dict(self.table.values)
            self.multiple.set(self.base['preview'].get('show_variants',False))
            self.full_frustums.set(self.base['preview'].get('variant_frustums',False))
            self.preview_limit.set(str(self.base['preview'].get('variant_limit',100)))
            shell_mode=self.base['preview']['shell_mode']
            self.shell_segment.set(self.base['preview'].get('shell_segment',shell_mode=='section'))
            self.section_angle.set(str(self.base['preview'].get('section_angle_deg',20)))
            self.mode.set(next(k for k,v in MODES.items() if v==('full' if shell_mode=='section' else shell_mode)))
            self.aim.set(next(k for k,v in AIMS.items() if v==self.base['parameterization']['camera_aim']))
            self.samples.set(str(self.base['render']['samples']))
            board_mode=self.base['render'].get('board_mode','diffuse')
            self.board_mode.set(next(label for label,value in BOARD_MODES.items() if value==board_mode))
            variant=self.base['render'].get('variant','scalar_rgb')
            self.backend.set(next((label for label,value in BACKENDS.items() if value==variant),'CPU'))
            from scene_model import Object3D
            sensor=next(obj.data['sensor'] for obj,_ in Object3D(self.base['graph']).walk() if obj.data['type']=='camera')
            self.width.set(str(round(sensor['resolution'][0]*self.base['render'].get('resolution_scale',1))))
            self.count.set(f'Комбинаций: {sweep_count(self.sweep,self.values):,}')
        finally:
            self.suppress=False
        self.queue_preview()

    def error(self,exc):
        self.status.set(str(exc))
        messagebox.showerror('Не получилось',str(exc),parent=self.main_window)

    def queue_autosave(self):
        if self.loading_state or self.suppress or self.closing:
            return
        self.state_dirty=True
        if self.autosave_job is not None:
            self.main_window.after_cancel(self.autosave_job)
        self.autosave_job=self.main_window.after(400,self.autosave_state)

    def autosave_state(self):
        if self.autosave_job is not None:
            self.main_window.after_cancel(self.autosave_job)
            self.autosave_job=None
        if not self.state_dirty:
            return
        try:
            values,sweep=self.table.snapshot()
            scene=resolve_scene(self.capture_base(),values)
            sweep_axes(sweep,values)
            scene['panel']={'pinned':self.pinned.get()}
            write_json(self.state_path,{'schema_version':1,'scene':scene,'sweep':sweep})
            self.state_dirty=False
        except (OSError,ValueError,TypeError) as exc:
            # Незавершённый ввод не должен затирать последний корректный набор.
            self.status.set(f'Автосохранение ожидает корректных значений: {exc}')

    def capture_base(self):
        # Копируем и вложенные словари, чтобы не менять исходную сцену.
        base=deepcopy(self.base)
        base['parameterization']['camera_aim']=AIMS[self.aim.get()]
        base['preview']['shell_mode']=MODES[self.mode.get()]
        base['preview']['shell_segment']=self.shell_segment.get()
        angle=float(self.section_angle.get().replace(',','.'))
        if not 1 <= angle <= 170:
            raise ValueError('Размер сегмента: от 1 до 170 градусов')
        base['preview']['section_angle_deg']=angle
        base['preview']['show_variants']=self.multiple.get()
        base['preview']['variant_frustums']=self.full_frustums.get()
        base['preview'].pop('full_display',None)
        base['preview']['variant_limit']=int(self.preview_limit.get())
        if not 1 <= base['preview']['variant_limit'] <= 500:
            raise ValueError('Лимит предпросмотра: от 1 до 500')
        samples=int(self.samples.get())
        width=int(self.width.get())
        if samples<1 or width<1:
            raise ValueError('Сэмплы и ширина должны быть положительными целыми')
        from scene_model import Object3D
        sensor=next(obj.data['sensor'] for obj,_ in Object3D(base['graph']).walk() if obj.data['type']=='camera')
        if width>sensor['resolution'][0]*4:
            raise ValueError('Ширина превышает исходное разрешение больше чем в 4 раза')
        base['render']['variant']=BACKENDS[self.backend.get()]
        base['render']['board_mode']=BOARD_MODES[self.board_mode.get()]
        base['render']['samples']=samples
        base['render']['resolution_scale']=width/sensor['resolution'][0]
        return base

    def table_changed(self):
        values,sweep=self.table.snapshot()
        resolve_scene(self.capture_base(),values)
        sweep_axes(sweep,values)
        self.values,self.sweep=values,sweep
        self.count.set(f'Комбинаций: {sweep_count(sweep,values):,}')
        self.queue_preview()

    def current_scene(self):
        self.table.finish_edit()
        self.values,self.sweep=self.table.snapshot()
        return resolve_scene(self.capture_base(),self.values)

    def queue_preview(self):
        self.queue_autosave()
        if self.suppress:
            return
        try:
            base=self.capture_base()
            scene=resolve_scene(base,self.values)
            self.pending_scene=(scene,deepcopy(self.sweep) if self.multiple.get() else None)
            import time
            self.pending_at=time.monotonic()+0.18
            self.preview_status.set('Ожидает обновления просмотра…')
        except Exception as exc:
            self.preview_status.set(f'Просмотр: {exc}')

    def collect_ranges(self):
        self.table.finish_edit()
        values,result=self.table.snapshot()
        sweep_axes(result,values)
        base=self.capture_base()
        for key,axis in sweep_axes(result,values).items():
            for value in [axis[0],axis[-1]]:
                resolve_scene(base,{**values,key:value})
        self.values=values
        return result

    def apply_ranges(self):
        try:
            sweep=self.collect_ranges()
            base=self.capture_base()
            base['parameters']=dict(self.values)
            self.sweep,self.base=sweep,base
            self.axes=sweep_axes(sweep,self.values)
            self.load_controls()
            self.status.set('Диапазоны применены. Рабочий набор сохраняется автоматически.')
            return True
        except Exception as exc:
            self.error(exc)
            return False

    def save_settings(self):
        if not self.apply_ranges():
            return
        try:
            scene=self.current_scene()
            write_json(ROOT/'scene.json',scene)
            write_json(ROOT/'sweep.json',self.sweep)
            self.base=scene
            self.status.set('Настройки сохранены в scene.json и sweep.json.')
        except Exception as exc:
            self.error(exc)

    def reload(self):
        try:
            base=read_json(ROOT/'scene.json'); sweep=read_json(ROOT/'sweep.json')
            resolve_scene(base); axes=sweep_axes(sweep,parameter_values(base))
            self.base,self.sweep,self.axes,self.values=base,sweep,axes,parameter_values(base)
            self.load_controls()
            self.status.set('JSON перечитан.')
        except Exception as exc:
            self.error(exc)

    def new_directory(self,prefix):
        return ROOT/'renders'/f'{prefix}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'

    def export_current(self):
        try:
            from scene_export import export_scene
            directory=self.new_directory('export')
            export_scene(self.current_scene(),directory)
            self.last_directory=directory
            self.status.set(f'Экспорт: {directory}')
            self.log_line(str(directory))
        except Exception as exc:
            self.error(exc)

    def export_cad(self):
        try:
            from scene_model import Scene
            directory=self.new_directory('cad')
            path=Scene.from_dict(self.current_scene()).export_step(directory/'scene.step')
            self.last_directory=directory
            self.status.set(f'Экспорт STEP и JSON: {directory}')
            self.log_line(str(path))
        except Exception as exc:
            self.error(exc)

    def start_job(self,batch):
        if self.job is not None:
            self.status.set('Уже идёт рендер. Дождитесь его или нажмите «Стоп».')
            return
        if not self.apply_ranges():
            return
        try:
            base=self.current_scene()
            sweep=deepcopy(self.sweep)
            # Для одного кадра задаём диапазоны, содержащие только текущее значение.
            if not batch:
                sweep['ranges']={key:{'min':value,'max':value,'step':1} for key,value in self.values.items()}
            directory=self.new_directory('batch' if batch else 'current')
            directory.mkdir(parents=True)
            write_json(directory/'source_scene.json',base)
            write_json(directory/'sweep.json',sweep)
            self.launch(directory)
        except Exception as exc:
            self.error(exc)

    def launch(self,directory):
        directory=Path(directory)
        self.last_directory=directory
        # Отдельный процесс использует тот же Python; -u отключает буферизацию
        # вывода, чтобы сообщения сразу появлялись в журнале.
        command=[sys.executable,'-u',str(ROOT/'scene_export.py'),'--scene',str(directory/'source_scene.json'),
                 '--sweep',str(directory/'sweep.json'),'--out',str(directory)]
        self.job=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        self.job_stopped=False
        self.status.set(f'Рендер: {directory}')
        self.log_line(f'Результаты: {directory}')
        stream=self.job.stdout
        # Чтение вывода может ждать новых строк, поэтому выносим его в отдельный поток.
        def read_output():
            for line in stream:
                self.job_queue.put(line.rstrip())
            stream.close()
        threading.Thread(target=read_output,daemon=True).start()

    def resume(self):
        if self.job is not None:
            self.status.set('Сначала остановите текущий рендер.')
            return
        directory=filedialog.askdirectory(parent=self.main_window,initialdir=ROOT/'renders',title='Папка эксперимента с source_scene.json и sweep.json')
        if not directory:
            return
        try:
            directory=Path(directory)
            read_json(directory/'source_scene.json'); read_json(directory/'sweep.json')
            self.launch(directory)
        except Exception as exc:
            self.error(exc)

    def stop_job(self):
        if self.job is not None:
            self.job_stopped=True
            self.job.terminate()
            self.status.set('Остановка; готовые кадры сохранены, эксперимент можно продолжить.')

    def log_line(self,text):
        self.log.configure(state='normal')
        self.log.insert('end',text+'\n')
        self.log.see('end')
        self.log.configure(state='disabled')

    def poll(self):
        """Обрабатывает результаты фоновых задач в главном потоке Tkinter."""
        import time
        while not self.job_queue.empty():
            self.log_line(self.job_queue.get_nowait())
        if self.job is not None and self.job.poll() is not None:
            code=self.job.returncode
            self.job=None
            self.status.set(('Остановлено; можно продолжить.' if self.job_stopped else
                             'Рендер завершён.' if code==0 else f'Ошибка рендера (код {code}); подробности ниже.')+f' {self.last_directory}')
        if self.preview_future is not None and self.preview_future.done():
            try:
                # Задача уже завершилась; result() передаст сюда возможную ошибку.
                self.preview_status.set(self.preview_future.result() or 'Просмотр обновлён.')
            except Exception as exc:
                self.preview_status.set(f'OCP Viewer: {exc}')
            self.preview_future=None
        if self.preview_future is None and self.pending_scene is not None and time.monotonic()>=self.pending_at:
            # Забираем последний запрос; новые изменения дождутся следующего обновления.
            (scene,sweep),self.pending_scene=self.pending_scene,None
            def display():
                if self.preview is None:
                    from scene_preview import Preview
                    self.preview=Preview()
                return self.preview.show(scene,sweep)
            self.preview_future=self.executor.submit(display)
            self.preview_status.set('Обновление просмотра…')
        if not self.closing:
            # Планируем следующий опрос через 100 мс, не блокируя цикл событий окна.
            self.poll_job=self.main_window.after(100,self.poll)

    def close(self):
        try:
            self.table.finish_edit()
        except ValueError:
            pass  # Предыдущий корректный набор остаётся сохранённым.
        self.autosave_state()
        self.closing=True
        self.main_window.after_cancel(self.poll_job)
        self.stop_job()
        self.executor.shutdown(wait=False,cancel_futures=True)
        self.main_window.destroy()


def main():
    main_window=tk.Tk()
    ScenePanel(main_window)
    # Цикл событий обрабатывает нажатия, перерисовку окна и вызовы after().
    main_window.mainloop()


if __name__=='__main__':
    main()
