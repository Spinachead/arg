import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from rapidocr_paddle import RapidOCR
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR

# 全局单例 OCR 实例，避免重复初始化
_ocr_instance = None

def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    global _ocr_instance
    if _ocr_instance is not None:
        return _ocr_instance
    
    # 设置 ONNX Runtime 线程数，避免与容器 CPU 限制冲突
    # 根据容器配置的 cpus: '16'，设置合理的线程数
    cpu_count = os.cpu_count() or 4
    intra_threads = min(4, cpu_count)  # 限制内部线程数
    inter_threads = min(2, cpu_count)  # 限制并行线程数
    
    os.environ.setdefault("OMP_NUM_THREADS", str(intra_threads))
    os.environ.setdefault("ONNXRUNTIME_CPU_NUM_THREADS", str(intra_threads))
    
    try:
        from rapidocr_paddle import RapidOCR

        _ocr_instance = RapidOCR(
            det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda
        )
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR

        # 显式设置线程数，避免 pthread_setaffinity_np 错误
        _ocr_instance = RapidOCR(
            intra_op_num_threads=intra_threads,
            inter_op_num_threads=inter_threads,
        )
    return _ocr_instance
