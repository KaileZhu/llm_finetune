import os
from modelscope import snapshot_download, AutoModel, AutoTokenizer


# 获取当前文件所在的目录
cache_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = snapshot_download('OpenBMB/MiniCPM-2B-sft-fp32', cache_dir=cache_dir, revision='master')