"""Shared ONNX Runtime CPU settings for both synthesis model families."""

import onnxruntime as ort


def make_session_options(intra_op_threads: int) -> ort.SessionOptions:
    """Create bounded-memory, sequential CPU options for one ONNX session."""
    if intra_op_threads < 1:
        raise ValueError("intra_op_threads must be positive")
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    return options
