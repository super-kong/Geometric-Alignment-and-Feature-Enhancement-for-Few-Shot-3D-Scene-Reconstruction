import torch
import gc


def clean_gpu_memory():

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("显存清理完成")

clean_gpu_memory()