import onnx.checker
import numpy as np
import torch.onnx
import torch.nn as nn
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

# 주어진 가중치
weights = {
    'w1': torch.tensor([[3.4243], [3.4299]], dtype=torch.float32),
    'b1': torch.tensor([-5.3119], dtype=torch.float32),
    'w2': torch.tensor([[4.4863], [4.4830]], dtype=torch.float32),
    'b2': torch.tensor([-1.7982], dtype=torch.float32),
    'w3': torch.tensor([[-7.1722], [6.7997]], dtype=torch.float32),
    'b3': torch.tensor([-3.0611], dtype=torch.float32),
}


# 사용자 정의 모델 클래스
class CustomModel(nn.Module):
    def __init__(self, weights):
        super(CustomModel, self).__init__()
        self.weights = weights

    def forward(self, x):
        f1 = torch.sigmoid(torch.matmul(x, self.weights['w1']) + self.weights['b1'])
        f2 = torch.sigmoid(torch.matmul(x, self.weights['w2']) + self.weights['b2'])
        output = torch.sigmoid(torch.matmul(torch.cat((f1, f2), 1), self.weights['w3']) + self.weights['b3'])
        return output


# 모델 인스턴스 생성
model = CustomModel(weights)

my_input = torch.tensor([[0.3456, 0.4032], [0.6356, 0.7032]], dtype=torch.float32)
my_input_label = torch.tensor([[0], [0]], dtype=torch.float32)
N = len(my_input)
model = BoundedModule(model, my_input)
ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
my_input = BoundedTensor(my_input, ptb)

pred = model(my_input)
label = [1 if pred[i] >= 0.5 else 0 for i in range(N)]
print(pred)
print(label)

for method in [
        'IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)',
        'CROWN-Optimized (alpha-CROWN)']:
    print('Bounding method:', method)
    if 'Optimized' in method:
        # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
        model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
    lb, ub = model.compute_bounds(x=(my_input,))
    for i in range(len(lb)):
        print(lb[i], "<=", "x", "<=", ub[i])

