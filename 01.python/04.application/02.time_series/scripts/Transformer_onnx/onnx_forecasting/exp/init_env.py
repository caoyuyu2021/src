# 获取项目路径
import os

def init_path():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../')) + "/"
    return path