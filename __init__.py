import sys
import os

# Добавляем путь расширения, чтобы Питон видел наши папки (r_facelib, scripts и т.д.)
repo_dir = os.path.dirname(os.path.realpath(__file__))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from .reactor_utils import init_trt_env
init_trt_env()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
