import torch
torch.cuda.empty_cache()  # 미사용 메모리 캐시 해제
torch.cuda.ipc_collect()  # PyTorch 내부 캐시 정리
import gc
gc.collect()  # 가비지 컬렉션 실행
torch.cuda.empty_cache()  # CUDA 캐시 비우기

# flake8: noqa
import os.path as osp

import hat.archs
import hat.data
import hat.models
from basicsr.test import test_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
