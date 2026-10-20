"""Запуск панели: bash 3d_scene/open_panel.sh.

Источник сцены — scene.json, диапазоны — sweep.json рядом с этим файлом.
Длины в JSON заданы в метрах, углы — в градусах.
"""
from scene_panel import main


if __name__ == '__main__':
    main()
