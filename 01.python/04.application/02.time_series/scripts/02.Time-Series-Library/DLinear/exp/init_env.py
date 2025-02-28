# 获取项目路径
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

def init_path():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../')) + '/'
    return path