"""Компактная редактируемая таблица и общий ползунок выбранного параметра."""
from copy import deepcopy
import tkinter as tk
from tkinter import ttk, font as tkfont

from scene_model import PARAMETERS, OFFSET_KEYS, range_values, parameter_values

GROUPS = {
    'Опорная камера': ['pitch_deg', 'yaw_deg', 'roll_deg'],
    'Шаблон': ['board_distance_m'],
    'Стекло': ['thickness_m', 'ior'],
    'Реальная камера — люфт': ['camera_dx_m', 'camera_dy_m', 'camera_dz_m',
                              'camera_ry_deg', 'camera_rz_deg', 'camera_rx_deg'],
}


class ParameterTable(ttk.Frame):
    def __init__(self, parent, on_change, on_error):
        super().__init__(parent)
        self.on_change, self.on_error = on_change, on_error
        self.values, self.ranges = {}, {}
        self.updating = False
        self.editor = None
        self.grid_lines=[]
        self.grid_pending=False
        self.grid_job=None
        self.current_labels={}
        self.current_font=tkfont.Font(root=self,font='TkDefaultFont')
        self.current_font.configure(weight='bold')
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        columns = ('enabled', 'preview_full', 'unit', 'current', 'min', 'max', 'step')
        self.tree = ttk.Treeview(self, columns=columns, show='tree headings', height=16, selectmode='browse')
        self.tree.heading('#0', text='Объект / параметр')
        self.tree.column('#0', width=250, minwidth=220, stretch=False)
        for key, title in zip(columns, ['Перебор', 'ППП', 'Ед.', 'ТЕКУЩЕЕ', 'От', 'До', 'Шаг']):
            self.tree.heading(key, text=title)
            self.tree.column(key, width=76, minwidth=55, anchor='center', stretch=True)
        self.tree.grid(row=0, column=0, sticky='nsew')
        self.tree.bind('<Destroy>',self.cancel_grid,add='+')
        scrollbar = ttk.Scrollbar(self, orient='vertical', command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        horizontal=ttk.Scrollbar(self,orient='horizontal',command=self.tree.xview)
        horizontal.grid(row=1,column=0,sticky='ew')
        def scroll_changed(scrollbar, first, last):
            scrollbar.set(first,last)
            self.queue_grid()
        self.tree.configure(yscrollcommand=lambda a,b:scroll_changed(scrollbar,a,b),
                            xscrollcommand=lambda a,b:scroll_changed(horizontal,a,b))
        for event in ('<Configure>','<B1-Motion>','<ButtonRelease-1>',
                      '<<TreeviewOpen>>','<<TreeviewClose>>'):
            self.tree.bind(event,self.queue_grid,add='+')
        for index, (label, keys) in enumerate(GROUPS.items()):
            group = f'group_{index}'
            self.tree.insert('', 'end', iid=group, text=label, open=True)
            for key in keys:
                self.tree.insert(group, 'end', iid=key, text=PARAMETERS[key].label)
        self.tree.bind('<Button-1>', self.click)
        self.tree.bind('<Double-1>', self.edit_cell)
        self.tree.bind('<space>', self.toggle_selected)
        self.tree.bind('<Shift-space>',lambda event:self.toggle_selected(flag='preview_full'))
        self.tree.bind('<<TreeviewSelect>>', lambda _: (self.update_slider(),self.queue_grid()))
        edit = ttk.Frame(self)
        edit.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(8,0))
        edit.columnconfigure(1, weight=1)
        self.selected_label = tk.StringVar(value='Выберите параметр')
        ttk.Label(edit, textvariable=self.selected_label, width=25).grid(row=0, column=0, sticky='w')
        self.slider = ttk.Scale(edit, from_=0, to=1, command=self.slide)
        self.slider.grid(row=0, column=1, sticky='ew', padx=8)
        self.current_text = tk.StringVar()
        entry = ttk.Entry(edit, textvariable=self.current_text, width=12)
        entry.grid(row=0, column=2)
        entry.bind('<Return>', lambda _: self.apply_selected())
        ttk.Button(edit, text='Применить', command=self.apply_selected).grid(row=0, column=3, padx=5)
        ttk.Label(self,text='ППП — полный перебор параметра при визуализации. Без галочки — только крайние значения.').grid(row=3,column=0,columnspan=2,sticky='w',pady=(5,0))
        ttk.Label(self,text='«Перебор» включает параметр в расчёт. Текущая сцена показывается всегда. Двойной щелчок — ввод.').grid(row=4,column=0,columnspan=2,sticky='w')

    def queue_grid(self,_=None):
        if not self.grid_pending:
            self.grid_pending=True
            self.grid_job=self.after_idle(self.draw_grid)

    def cancel_grid(self,_=None):
        if self.grid_job is not None:
            self.after_cancel(self.grid_job)
            self.grid_job=None
        self.grid_pending=False

    def draw_grid(self):
        self.grid_job=None
        self.grid_pending=False
        if not self.tree.winfo_exists():
            return
        rows=[]
        for group in self.tree.get_children():
            for item in (group,*self.tree.get_children(group)):
                box=self.tree.bbox(item)
                if box:
                    rows.append((item,box))
        segments=[]
        if rows:
            width,height=self.tree.winfo_width(),self.tree.winfo_height()
            top=max(1,rows[0][1][1])
            bottom=min(height-1,max(box[1]+box[3] for _,box in rows))
            segments.append((1,top,width-2,1))
            for _,(_,y,_,row_height) in rows:
                edge=y+row_height-1
                if top <= edge < height:
                    segments.append((1,edge,width-2,1))
            for column in ['#0',*self.tree['columns']]:
                box=self.tree.bbox(rows[0][0],column)
                if box:
                    x=box[0]+box[2]-1
                    if 0 < x < width-1:
                        segments.append((x,top,1,max(1,bottom-top)))
        while len(self.grid_lines)<len(segments):
            line=tk.Frame(self.tree,background='#b5bdc6',borderwidth=0,
                          highlightthickness=0,takefocus=False)
            self.grid_lines.append(line)
        for index,line in enumerate(self.grid_lines):
            if index<len(segments):
                x,y,width,height=segments[index]
                line.place(x=x,y=y,width=width,height=height)
                line.lift()
            else:
                line.place_forget()
        self.draw_current_values(rows)
        if self.editor is not None:
            self.editor.lift()

    def draw_current_values(self,rows):
        visible={key for key,_ in rows if key in PARAMETERS}
        style=ttk.Style(self)
        selected=set(self.tree.selection())
        for key in visible:
            box=self.tree.bbox(key,'current')
            if not box:
                continue
            if key not in self.current_labels:
                label=tk.Label(self.tree,font=self.current_font,borderwidth=0,
                               highlightthickness=0,padx=0,pady=0,takefocus=False)
                label.bind('<Button-1>',lambda event,k=key:self.select_current(k))
                label.bind('<Double-1>',lambda event,k=key:self.edit_current(k))
                label.bind('<MouseWheel>',lambda event:self.tree.event_generate('<MouseWheel>',delta=event.delta))
                for button in (4,5):
                    label.bind(f'<Button-{button}>',lambda event,b=button:self.tree.event_generate(f'<Button-{b}>'))
                self.current_labels[key]=label
            label=self.current_labels[key]
            state=('selected',) if key in selected else ()
            background=style.lookup('Treeview','background',state) or 'white'
            foreground=style.lookup('Treeview','foreground',state) or 'black'
            label.configure(text=self.tree.set(key,'current'),background=background,foreground=foreground)
            x,y,width,height=box
            label.place(x=x+1,y=y+1,width=max(1,width-2),height=max(1,height-2))
            label.lift()
        for key,label in self.current_labels.items():
            if key not in visible:
                label.place_forget()

    def select_current(self,key):
        self.tree.focus_set()
        self.tree.focus(key)
        self.tree.selection_set(key)

    def edit_current(self,key):
        from types import SimpleNamespace
        self.select_current(key)
        box=self.tree.bbox(key,'current')
        if box:
            self.edit_cell(SimpleNamespace(x=box[0]+box[2]//2,y=box[1]+box[3]//2))
        return 'break'

    def resize_columns(self):
        font=tkfont.nametofont('TkDefaultFont',root=self)
        attributes=font.actual()
        attributes['weight']='bold'
        self.current_font.configure(**attributes)
        labels=[spec.label for spec in PARAMETERS.values()]+list(GROUPS)
        width=max(font.measure(label) for label in labels)+44
        self.tree.column('#0',width=width,minwidth=120,stretch=False)
        heading_font=tkfont.nametofont('TkHeadingFont',root=self)
        for key in ('enabled','preview_full','unit','current','min','max','step'):
            texts=[self.tree.heading(key,'text')]
            texts += [self.tree.set(item,key) for item in PARAMETERS]
            cell_font=self.current_font if key=='current' else font
            width=max(heading_font.measure(texts[0]),*(cell_font.measure(text) for text in texts[1:]))
            padding=28 if key=='current' else 14
            width+=padding
            self.tree.column(key,width=width,minwidth=width,stretch=False)
        self.queue_grid()

    def load(self, scene, sweep):
        self.cancel_edit()
        self.values = parameter_values(scene)
        self.ranges = deepcopy(sweep['ranges'])
        for key in PARAMETERS:
            if key not in self.ranges:
                value = self.values[key]
                step = .01 if key.endswith('_m') else 1
                self.ranges[key] = dict(min=value, max=value, step=step, enabled=False)
            self.ranges[key].setdefault('preview_full',scene['preview'].get('full_display',False))
            self.refresh_row(key)
        if self.selected_key() is None:
            self.tree.selection_set('pitch_deg')
        self.update_slider()
        self.resize_columns()

    def selected_key(self):
        selected = self.tree.selection()
        return selected[0] if selected and selected[0] in PARAMETERS else None

    def refresh_row(self, key):
        spec, factor = self.ranges[key], PARAMETERS[key].display_factor
        self.tree.item(key, values=(
            '☑' if spec.get('enabled', True) else '☐',
            '☑' if spec.get('preview_full', False) else '☐', PARAMETERS[key].unit,
            f'{self.values[key]*factor:g}',
            *(f'{spec[name]*factor:g}' for name in ['min', 'max', 'step'])))
        self.queue_grid()

    def snapshot(self):
        sweep = {'schema_version': 1, 'ranges': deepcopy(self.ranges)}
        for key, spec in sweep['ranges'].items():
            spec['value'] = self.values[key]
        return dict(self.values), sweep

    def toggle(self,key,flag='enabled'):
        self.ranges[key][flag]=not self.ranges[key].get(flag,flag=='enabled')
        self.refresh_row(key)
        self.on_change()

    def toggle_selected(self,_=None,flag='enabled'):
        key = self.selected_key()
        if key:
            self.toggle(key,flag)
        return 'break'

    def click(self, event):
        key = self.tree.identify_row(event.y)
        column=self.tree.identify_column(event.x)
        flags={'#1':'enabled','#2':'preview_full'}
        if key in PARAMETERS and column in flags:
            self.tree.selection_set(key)
            self.toggle(key,flags[column])
            return 'break'

    def cancel_edit(self):
        if self.editor is not None:
            editor, self.editor = self.editor, None
            editor.destroy()

    def edit_cell(self, event):
        key,column=self.tree.identify_row(event.y),self.tree.identify_column(event.x)
        if key not in PARAMETERS or not column or column=='#0':
            return 'break'
        name=self.tree['columns'][int(column[1:])-1]
        if name not in ('current','min','max','step'):
            return 'break'
        self.cancel_edit()
        box = self.tree.bbox(key, column)
        if not box:
            return 'break'
        value = self.values[key] if name == 'current' else self.ranges[key][name]
        editor = ttk.Entry(self.tree)
        self.editor = editor
        editor.insert(0, f'{value*PARAMETERS[key].display_factor:g}')
        editor.place(x=box[0], y=box[1], width=box[2], height=box[3])
        editor.focus_set()
        editor.select_range(0, 'end')
        def commit(_=None):
            if self.editor is not editor:
                return
            if self.set_number(key, name, editor.get()):
                self.cancel_edit()
        editor.bind('<Return>', commit)
        editor.bind('<FocusOut>', commit)
        editor.bind('<Escape>', lambda _: self.cancel_edit())
        return 'break'

    def finish_edit(self):
        if self.editor is not None:
            self.editor.event_generate('<Return>')
            if self.editor is not None:
                raise ValueError('Исправьте значение в редактируемой ячейке')

    def set_number(self, key, name, text):
        old_values, old_ranges = deepcopy(self.values), deepcopy(self.ranges)
        try:
            value = float(text.replace(',', '.'))/PARAMETERS[key].display_factor
            import math
            if not math.isfinite(value):
                raise ValueError('Введите конечное число')
            if name == 'current':
                self.values[key] = value
            else:
                self.ranges[key][name] = value
                range_values(self.ranges[key])
            self.on_change()
        except Exception as exc:
            self.values, self.ranges = old_values, old_ranges
            self.on_error(exc)
            return False
        self.refresh_row(key)
        for column in ('current','min','max','step'):
            font=self.current_font if column=='current' else tkfont.nametofont('TkDefaultFont',root=self)
            needed=font.measure(self.tree.set(key,column))+14
            if needed>self.tree.column(column,'width'):
                self.tree.column(column,width=needed)
        self.update_slider()
        return True

    def update_slider(self):
        key = self.selected_key()
        self.updating = True
        try:
            if key is None or key not in self.ranges:
                self.slider.configure(state='disabled')
                return
            self.selected_label.set(f'{PARAMETERS[key].label}, {PARAMETERS[key].unit}')
            axis = range_values(self.ranges[key])
            self.slider.configure(from_=0, to=max(1,len(axis)-1), state='normal' if len(axis)>1 else 'disabled')
            self.slider.set(min(range(len(axis)), key=lambda i: abs(axis[i]-self.values[key])))
            self.current_text.set(f'{self.values[key]*PARAMETERS[key].display_factor:g}')
        finally:
            self.updating = False

    def slide(self, value):
        key = self.selected_key()
        if self.updating or key is None:
            return
        axis = range_values(self.ranges[key])
        index = max(0, min(len(axis)-1, round(float(value))))
        text = f'{axis[index]*PARAMETERS[key].display_factor:.15g}'
        self.set_number(key, 'current', text)

    def apply_selected(self):
        key = self.selected_key()
        if key:
            self.set_number(key, 'current', self.current_text.get())
