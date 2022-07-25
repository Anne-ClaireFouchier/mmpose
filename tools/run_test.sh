"""
# python test.py --out filename.json

need to add --gpu-id if i want to choose a GPU, otherwise default 0 (and --launcher none but is none by default)
--out filename.json 


use single_gpu_test from mmpose.apis


not sure if i should use --gpu-collect : gpu to collect results

check dataset.evaluate
from mmpose.datasets import build_dataloader, build_dataset
"""

python my_test.py 'config' 'checkpoint' --out filename.json '--work-dir' dir_to_save_results 
