"""Проверка создания настоящих виджетов панели; требует графического дисплея."""
from pathlib import Path
import sys
import tkinter as tk
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene_model import PARAMETERS
from scene_panel import ScenePanel


class PanelTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from shutil import copyfile
        from unittest.mock import patch
        import scene_panel
        from scene_model import ROOT
        directory=tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        target=Path(directory.name)
        for name in ['scene.json','sweep.json']:
            copyfile(ROOT/name,target/name)
        patcher=patch.object(scene_panel,'ROOT',target)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_working_state_restores_without_explicit_save(self):
        import scene_panel
        from scene_model import read_json
        try:
            root=tk.Tk()
        except tk.TclError as exc:
            if 'display' in str(exc).lower():
                self.skipTest(str(exc))
            raise
        root.withdraw()
        panel=ScenePanel(root)
        try:
            panel.table.set_number('camera_dx_m','current','1.5')
            panel.table.set_number('camera_dx_m','min','-2')
            panel.table.set_number('camera_dx_m','max','3')
            panel.table.toggle('yaw_deg')
            panel.multiple.set(True)
            panel.full_frustums.set(True)
            panel.table.toggle('camera_dx_m',flag='preview_full')
            panel.queue_preview()
            panel.shell_segment.set(True)
            panel.section_angle.set('30')
            panel.backend.set('GPU NVIDIA (CUDA)')
            panel.board_mode.set('Светящаяся')
            panel.width.set('320')
            panel.pinned.set(False)
            # Завершение отложенного after без кнопки «Сохранить настройки».
            panel.pending_scene=None
            root.after(600,root.quit)
            root.mainloop()
            saved=read_json(panel.state_path)
            self.assertEqual(saved['scene']['parameters']['camera_dx_m'],.015)
            self.assertFalse(saved['sweep']['ranges']['yaw_deg']['enabled'])
            self.assertFalse(read_json(scene_panel.ROOT/'scene.json')['parameters'].get('camera_dx_m',0))
            # При закрытии сохраняется и последняя правка до истечения задержки.
            panel.table.set_number('camera_dy_m','current','0.5')
        finally:
            panel.close()
        root=tk.Tk();root.withdraw()
        restored=ScenePanel(root)
        try:
            self.assertEqual(restored.values['camera_dx_m'],.015)
            self.assertEqual(restored.values['camera_dy_m'],.005)
            self.assertFalse(restored.table.ranges['yaw_deg']['enabled'])
            self.assertEqual(restored.table.ranges['camera_dx_m']['min'],-.02)
            self.assertEqual(restored.table.ranges['camera_dx_m']['max'],.03)
            self.assertTrue(restored.multiple.get())
            self.assertTrue(restored.full_frustums.get())
            self.assertTrue(restored.table.ranges['camera_dx_m']['preview_full'])
            self.assertFalse(restored.table.ranges['camera_dy_m']['preview_full'])
            self.assertTrue(restored.shell_segment.get())
            self.assertEqual(float(restored.section_angle.get()),30)
            self.assertEqual(restored.board_mode.get(),'Светящаяся')
            self.assertEqual(restored.capture_base()['render']['board_mode'],'emissive')
            self.assertEqual(restored.backend.get(),'GPU NVIDIA (CUDA)')
            self.assertEqual(restored.capture_base()['render']['variant'],'cuda_ad_rgb')
            self.assertEqual(restored.width.get(),'320')
            self.assertFalse(restored.pinned.get())
            # Некорректный незавершённый ввод не портит файл.
            before=restored.state_path.read_bytes()
            restored.samples.set('')
            restored.autosave_state()
            self.assertEqual(restored.state_path.read_bytes(),before)
        finally:
            root.update_idletasks()
            restored.close()

    def test_font_size_persists_and_updates_table(self):
        import tempfile
        from shutil import copyfile
        from unittest.mock import patch
        import tkinter.font as tkfont
        import scene_panel
        from scene_model import ROOT,read_json
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)
            for name in ['scene.json','sweep.json']:
                copyfile(ROOT/name,target/name)
            with patch.object(scene_panel,'ROOT',target):
                for expected in [None,14]:
                    try:
                        root=tk.Tk()
                    except tk.TclError as exc:
                        if 'display' in str(exc).lower():
                            self.skipTest(str(exc))
                        raise
                    root.withdraw()
                    panel=None
                    try:
                        panel=ScenePanel(root)
                        if expected is not None:
                            self.assertEqual(panel.font_size.get(),str(expected))
                        # Настройка находится в отдельной верхней строке окна.
                        toolbar=panel.font_entry.master
                        self.assertEqual(int(toolbar.grid_info()['row']),0)
                        self.assertIs(toolbar.master,root)
                        old_width=panel.table.tree.column('#0','width')
                        panel.font_size.set('14')
                        panel.apply_font_size()
                        self.assertEqual(tkfont.nametofont('TkDefaultFont',root=root).cget('size'),14)
                        self.assertEqual(read_json(target/'ui_settings.json'),{'font_size':14})
                        self.assertFalse(panel.table.tree.column('#0','stretch'))
                        self.assertGreaterEqual(panel.table.tree.column('#0','width'),old_width)
                        self.assertGreater(int(tk.ttk.Style(root).lookup('Treeview','rowheight')),14)
                        panel.font_size.set('0')
                        panel.apply_font_size()
                        self.assertEqual(panel.font_size.get(),'14')
                        self.assertEqual(read_json(target/'ui_settings.json')['font_size'],14)
                        # Настоящее действие Spinbox, а не прямой вызов обработчика.
                        root.update_idletasks()
                        panel.font_entry.event_generate('<<Increment>>')
                        self.assertEqual(panel.font_size.get(),'15')
                        self.assertEqual(read_json(target/'ui_settings.json')['font_size'],15)
                        panel.font_entry.event_generate('<<Decrement>>')
                        self.assertEqual(panel.font_size.get(),'14')
                    finally:
                        if panel is not None:
                            root.update_idletasks()
                            panel.close()
                        else:
                            root.destroy()

    def test_panel_initialization(self):
        try:
            main_window = tk.Tk()
        except tk.TclError as exc:
            if 'display' in str(exc).lower():
                self.skipTest(f'Нет доступа к графическому дисплею: {exc}')
            raise
        main_window.withdraw()
        panel = None
        try:
            panel = ScenePanel(main_window)
            main_window.update_idletasks()
            self.assertEqual(set(panel.table.values), set(PARAMETERS))
            self.assertIsNotNone(panel.pending_scene)
            self.assertTrue(panel.apply_ranges())
            self.assertTrue(panel.table.set_number('board_distance_m','current','230'))
            self.assertTrue(panel.table.set_number('camera_dx_m','current','1.5'))
            self.assertAlmostEqual(panel.values['board_distance_m'],2.3)
            scene=panel.current_scene()
            self.assertAlmostEqual(scene['parameterization']['camera_offset']['translation_m'][0],.015)
            self.assertEqual(panel.table.tree.set('thickness_m','current'),'5')
            # Выключение сохраняет диапазон, но фиксирует текущее значение.
            previous=dict(panel.table.ranges['yaw_deg'])
            panel.table.toggle('yaw_deg')
            self.assertFalse(panel.sweep['ranges']['yaw_deg']['enabled'])
            self.assertEqual(panel.sweep['ranges']['yaw_deg']['max'],previous['max'])
            panel.table.tree.selection_set('pitch_deg')
            panel.table.update_slider()
            panel.table.slide('0')
            self.assertEqual(panel.values['pitch_deg'],40)
            self.assertTrue(panel.apply_ranges())
            self.assertEqual(panel.values['pitch_deg'],40)
            # Ошибочный ввод отклоняется, прежнее значение остаётся.
            self.assertFalse(panel.table.set_number('thickness_m','current','-1'))
            self.assertEqual(panel.values['thickness_m'],.05)
            panel.multiple.set(True)
            panel.queue_preview()
            self.assertIsNotNone(panel.pending_scene[1])
            # Сохранение и перечитывание не теряют значения и галочки.
            import tempfile
            from unittest.mock import patch
            import scene_panel
            from scene_model import read_json
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(scene_panel,'ROOT',Path(directory)):
                    panel.save_settings()
                    saved=read_json(Path(directory)/'sweep.json')
                    self.assertFalse(saved['ranges']['yaw_deg']['enabled'])
                    self.assertAlmostEqual(saved['ranges']['camera_dx_m']['value'],.015)
                    panel.table.set_number('camera_dx_m','current','0')
                    panel.reload()
                    self.assertAlmostEqual(panel.values['camera_dx_m'],.015)
                    self.assertTrue(panel.multiple.get())
        finally:
            if panel is not None:
                panel.close()
            else:
                main_window.destroy()


if __name__ == '__main__':
    unittest.main()
