import pytest


def test_runtime_options_limit_threads_for_both_model_families() -> None:
    from itantra_tts.adapters.ort_runtime import make_session_options

    options = make_session_options(2)

    assert options.intra_op_num_threads == 2
    assert options.inter_op_num_threads == 1
    assert options.enable_cpu_mem_arena is False
    assert options.enable_mem_pattern is False


def test_runtime_options_reject_zero_threads() -> None:
    from itantra_tts.adapters.ort_runtime import make_session_options

    with pytest.raises(ValueError, match="positive"):
        make_session_options(0)
