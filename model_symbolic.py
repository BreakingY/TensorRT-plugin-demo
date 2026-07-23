import torch
import torch.nn as nn
from torch.autograd import Function

import onnx

from cuda_op.centerpool import *



# ============================================================
# 1. CUDA CenterPool 包装
# ============================================================
class CenterPoolFunction(Function):

    @staticmethod
    def forward(ctx, x):
        y = centerpool(x)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        return None

    @staticmethod
    def symbolic(g, x):
        return g.op("mydomain::CenterPool", x)

class Model(nn.Module):
    def forward(self, x):
        return CenterPoolFunction.apply(x)

def print_tensor_shape(value_info):
    print("name:", value_info.name)
    if not value_info.type.HasField("tensor_type"):
        print("  tensor type unknown")
        return

    tensor_type = value_info.type.tensor_type

    if not tensor_type.HasField("shape"):
        print("  shape unknown")
        return

    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            print(" ", dim.dim_value)
        elif dim.HasField("dim_param"):
            print(" ", dim.dim_param)
        else:
            print(" unknown")

if __name__ == "__main__":
    torch.cuda.set_device(2)

    print("==============================")
    print("测试 CUDA CenterPool")
    print("==============================")

    model = Model().cuda()
    model.eval()
    x_test = torch.ones(1, 64, 80, 80).cuda() * 0.5

    with torch.no_grad():
        y = model(x_test)

    print("模型测试输出:")
    print(y)

    print("\n==============================")
    print("导出 ONNX")
    print("==============================")
    x = torch.randn(1, 64, 80, 80).cuda()

    torch.onnx.export(
        model,
        x,
        "model_symbolic.onnx",
        opset_version=11,
        input_names=[
            "input"
        ],
        output_names=[
            "output"
        ],
        verbose=False
    )
    print("已导出 model_symbolic.onnx")

    model_onnx = onnx.load("model_symbolic.onnx")

    print("\n==============================")
    print("ONNX节点")
    print("==============================")

    for node in model_onnx.graph.node:
        domain = (
            node.domain
            if node.domain
            else "(default)"
        )
        print(f"domain={domain}, op_type={node.op_type}")


    print("\n==============================")
    print("输入")
    print("==============================")

    for inp in model_onnx.graph.input:
        print_tensor_shape(inp)

    print("\n==============================")
    print("输出")
    print("==============================")
    for out in model_onnx.graph.output:
        print_tensor_shape(out)

    print("\n==============================")
    print("ONNX checker")
    print("==============================")

    try:
        onnx.checker.check_model(model_onnx)
        print("ONNX 模型结构检查通过!")
    except Exception as e:
        print("ONNX模型检查失败:")
        print(e)
