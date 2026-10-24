from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from tools.experimental_fastpitch_int4_storage import convert


ROOT = Path(__file__).resolve().parents[2]
HI_FASTPITCH = ROOT / "models/artifacts/echobharat-int8/hi/fastpitch-hi.int8.onnx"


@pytest.mark.skipif(not HI_FASTPITCH.is_file(), reason="research model not provisioned")
def test_hindi_fastpitch_storage_conversion_is_executable(tmp_path: Path) -> None:
    output = tmp_path / "fastpitch-hi.int4-storage.experimental.onnx"
    count = convert(HI_FASTPITCH, output)
    assert count == 56
    graph = onnx.load(output)
    assert sum(item.data_type == onnx.TensorProto.INT4 for item in graph.graph.initializer) == count
    assert output.stat().st_size < HI_FASTPITCH.stat().st_size
    onnx.checker.check_model(graph)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(output), opts, providers=["CPUExecutionProvider"])
    mel = session.run(None, {"text": np.array([[1, 2, 3, 4]], dtype=np.int64)})[0]
    assert mel.size > 0
    assert np.isfinite(mel).all()
    with pytest.raises(ValueError, match="new file"):
        convert(HI_FASTPITCH, output)
