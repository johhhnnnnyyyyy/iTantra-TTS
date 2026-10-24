"""Conservative NNAPI readiness check for the derived Odia artifact.

This is deliberately a gate, not a performance estimator.  A result of
``mobile_ready`` means only that the generated required-operator contract has
no known NNAPI-incompatible operators; it still needs an instrumented Android
run on the target device before it can be called supported.
"""

from dataclasses import dataclass


# This is the subset relevant to this model from ONNX Runtime's Android NNAPI
# supported-operators table.  Operators outside it may run on the CPU EP, so
# their presence prevents a claim of full NNAPI delegation.
_NNAPI_OPERATORS = frozenset(
    {
        "ai.onnx:Add",
        "ai.onnx:Cast",
        "ai.onnx:Concat",
        "ai.onnx:DequantizeLinear",
        "ai.onnx:Div",
        "ai.onnx:Exp",
        "ai.onnx:Gather",
        "ai.onnx:MatMul",
        "ai.onnx:Mul",
        "ai.onnx:Neg",
        "ai.onnx:Pad",
        "ai.onnx:Pow",
        "ai.onnx:ReduceMean",
        "ai.onnx:Relu",
        "ai.onnx:Reshape",
        "ai.onnx:Sigmoid",
        "ai.onnx:Slice",
        "ai.onnx:Softmax",
        "ai.onnx:Split",
        "ai.onnx:Sqrt",
        "ai.onnx:Squeeze",
        "ai.onnx:Sub",
        "ai.onnx:Tanh",
        "ai.onnx:Transpose",
        "ai.onnx:Unsqueeze",
    }
)


@dataclass(frozen=True)
class OdiaMobileReadiness:
    """Required-operator result for a concrete converted ORT model."""

    required_operators: tuple[str, ...]
    nnapi_blockers: tuple[str, ...]

    @property
    def mobile_ready(self) -> bool:
        """Whether the graph has no known full-NNAPI-delegation blockers."""
        return not self.nnapi_blockers


def analyze_required_operators(config_text: str) -> OdiaMobileReadiness:
    """Parse ORT's generated config and identify operators outside NNAPI's subset."""
    required: list[str] = []
    for raw_line in config_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        domain, _, operators = line.split(";", maxsplit=2)
        for item in _split_operator_items(operators):
            operator = item.split("{", maxsplit=1)[0]
            required.append(f"{domain}:{operator}")

    unique_required = tuple(sorted(set(required)))
    return OdiaMobileReadiness(
        required_operators=unique_required,
        nnapi_blockers=tuple(
            operator for operator in unique_required if operator not in _NNAPI_OPERATORS
        ),
    )


def _split_operator_items(operators: str) -> tuple[str, ...]:
    """Split config operators on commas that are outside type JSON metadata."""
    items: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(operators):
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
        elif character == "," and depth == 0:
            items.append(operators[start:index])
            start = index + 1
    items.append(operators[start:])
    return tuple(item.strip() for item in items if item.strip())
