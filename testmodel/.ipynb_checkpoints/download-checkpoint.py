import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('vacant1895/SpringFestQA-ESFP', cache_dir='./')
