import ast
import operator as op
from typing import Any, Optional, Tuple

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class _SafeEval:
    """Safely evaluate a simple arithmetic expression using ast.

    Supported: integers/floats, + - * / // % **, parentheses, unary +/-, and spaces.
    """

    _ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv,
        ast.Mod: op.mod,
        ast.Pow: op.pow,
        ast.UAdd: op.pos,
        ast.USub: op.neg,
    }

    @classmethod
    def eval(cls, expr: str) -> float:
        node = ast.parse(expr, mode="eval")
        return cls._eval(node.body)

    @classmethod
    def _eval(cls, node):
        if isinstance(node, ast.Num):  # py<3.8 compatibility
            return node.n
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are allowed")
        if isinstance(node, ast.BinOp) and type(node.op) in cls._ops:
            return cls._ops[type(node.op)](cls._eval(node.left), cls._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in cls._ops:
            return cls._ops[type(node.op)](cls._eval(node.operand))
        if isinstance(node, ast.Expr):
            return cls._eval(node.value)
        if isinstance(node, ast.Paren):  # not used in recent Python ASTs
            return cls._eval(node.value)
        raise ValueError("Unsupported expression")


class CalculatorTool(BaseTool):
    """A simple in-process calculator tool.

    Parameters:
        expression (str): arithmetic expression to evaluate, e.g. "12 + 7 * (3 - 1)".
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        # accept provided tool_schema or default to a sensible schema
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema.model_validate(
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "Evaluate a basic arithmetic expression.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Arithmetic expression to evaluate (e.g., '3*(4+5)').",
                                }
                            },
                            "required": ["expression"],
                        },
                    },
                }
            )
        super().__init__(config, tool_schema)
        self._state = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> Tuple[str, ToolResponse]:
        # Create per-trajectory instance if needed
        if instance_id is None:
            instance_id = "calculator"
        self._state[instance_id] = {}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[ToolResponse, float, dict]:
        expr = parameters.get("expression", "")
        try:
            value = _SafeEval.eval(str(expr))
            return ToolResponse(text=str(value)), None, None
        except Exception as e:
            return ToolResponse(text=f"error: {e}"), None, None

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        # No internal reward; return empty placeholder.
        return ""

    async def release(self, instance_id: str, **kwargs) -> None:
        self._state.pop(instance_id, None)

