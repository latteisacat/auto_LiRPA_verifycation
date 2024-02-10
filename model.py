import onnx.checker
import numpy as np
import torch.onnx
import torch.nn as nn
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from collections import defaultdict

#주어진 가중치
weights = {
    'w1': torch.tensor([[3.4243], [3.4299]], dtype=torch.float32),
    'b1': torch.tensor([-5.3119], dtype=torch.float32),
    'w2': torch.tensor([[4.4863], [4.4830]], dtype=torch.float32),
    'b2': torch.tensor([-1.7982], dtype=torch.float32),
    'w3': torch.tensor([[-7.1722], [6.7997]], dtype=torch.float32),
    'b3': torch.tensor([-3.0611], dtype=torch.float32),
}


# 사용자 정의 모델 클래스 v1
class CustomModel_old(nn.Module):
    def __init__(self, weights):
        super(CustomModel_old, self).__init__()
        self.weights = weights

    def forward(self, x):
        f1 = torch.sigmoid(torch.matmul(x, self.weights['w1']) + self.weights['b1'])
        f2 = torch.sigmoid(torch.matmul(x, self.weights['w2']) + self.weights['b2'])
        output = torch.sigmoid(torch.matmul(torch.cat((f1, f2), 1), self.weights['w3']) + self.weights['b3'])
        return output


class CustomOutputLayer(nn.Module):
    def __init__(self):
        super(CustomOutputLayer, self).__init__()

    def forward(self, x):
        # 이 값이 0.5보다 크면 0으로 취급
        output1 = 1 - x
        # 이 값이 0.5보다 크면 1로 취급
        output2 = x
        # 두 출력을 하나로 합치기
        combined_output = torch.cat((output1, output2), dim=1)
        return combined_output


# 사용자 정의 모델 클래스 v2
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.weights = weights
        self.fc1 = nn.Linear(2, 2)  # 입력층과 은닉층 사이의 연결
        self.fc2 = nn.Linear(2, 1)  # 은닉층과 출력층 사이의 연결
        self.fc3 = CustomOutputLayer()  # 검사를 위해 새로운 출력층 추가(label)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # 은닉층에 시그모이드 활성화 함수 적용
        x = torch.sigmoid(self.fc2(x))  # 출력층에 시그모이드 활성화 함수 적용
        x = self.fc3(x)  # 활성화 함수 없음, 아무 계산 없이 값 전달
        return x


# 모델 인스턴스 생성
model = CustomModel()
# model = CustomModel_old(weights)
# 미리 정해진 가중치로 설정
with torch.no_grad():  # 가중치 설정 시에는 기울기를 추적할 필요가 없으므로 기울기 추적을 비활성화합니다.
    model.fc1.weight = nn.Parameter(torch.tensor([[3.4243, 3.4299], [4.4863, 4.4830]]))  # 첫 번째 레이어의 가중치를 설정합니다.
    model.fc1.bias = nn.Parameter(torch.tensor([-5.3119, -1.7982]))  # 첫 번째 레이어의 편향을 설정합니다.
    model.fc2.weight = nn.Parameter(torch.tensor([[-7.1722, 6.7997]]))  # 두 번째 레이어의 가중치를 설정합니다.
    model.fc2.bias = nn.Parameter(torch.tensor([-3.0611]))  # 두 번째 레이어의 편향을 설정합니다.

my_input = torch.tensor([
    [0.3456, 0.4032],
    [0.6356, 0.7032],
    [0.1101, 0.1234],
    [0.9393, 0.1102],
    [0.4002, 0.5567],
    [0.2234, 0.7654]
], dtype=torch.float32)
my_input_label = torch.tensor([[0], [0], [0], [1], [1], [1]], dtype=torch.int64)
N = len(my_input)
n_classes = 2  # true(1) or false(0)의 2 개의 label을 가짐
model = BoundedModule(model, my_input)
ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
my_input = BoundedTensor(my_input, ptb)

pred = model(my_input)
label = [1 if pred[i][0] < pred[i][1] else 0 for i in range(N)]
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
    for i in range(N):
        print(f'input {i} top-1 prediction {label[i]} ground-truth {my_input_label[i].item()}')
        for j in range(n_classes):
            indicator = '(ground-truth)' if j == my_input_label[i] else ''
            print('f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}'.format(
                j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator))
    print()


print('Demonstration 2: Obtaining linear coefficients of the lower and upper bounds.\n')
# There are many bound coefficients during CROWN bound calculation; here we are interested in the linear bounds
# of the output layer, with respect to the input layer (the image).
required_A = defaultdict(set)
required_A[model.output_name[0]].add(model.input_name[0])

for method in [
        'IBP+backward (CROWN-IBP)', 'backward (CROWN)', 'CROWN',
        'CROWN-Optimized (alpha-CROWN)']:
    print("Bounding method:", method)
    if 'Optimized' in method:
        # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
        model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
    lb, ub, A_dict = model.compute_bounds(x=(my_input,), method=method.split()[0], return_A=True, needed_A_dict=required_A)
    lower_A, lower_bias = A_dict[model.output_name[0]][model.input_name[0]]['lA'], A_dict[model.output_name[0]][model.input_name[0]]['lbias']
    upper_A, upper_bias = A_dict[model.output_name[0]][model.input_name[0]]['uA'], A_dict[model.output_name[0]][model.input_name[0]]['ubias']
    print(f'lower bound linear coefficients size (batch, output_dim, *input_dims): {list(lower_A.size())}')
    print(f'lower bound linear coefficients norm (smaller is better): {lower_A.norm()}')
    print(f'lower bound bias term size (batch, output_dim): {list(lower_bias.size())}')
    print(f'lower bound bias term sum (larger is better): {lower_bias.sum()}')
    print(f'upper bound linear coefficients size (batch, output_dim, *input_dims): {list(upper_A.size())}')
    print(f'upper bound linear coefficients norm (smaller is better): {upper_A.norm()}')
    print(f'upper bound bias term size (batch, output_dim): {list(upper_bias.size())}')
    print(f'upper bound bias term sum (smaller is better): {upper_bias.sum()}')
    print(f'These linear lower and upper bounds are valid everywhere within the perturbation radii.\n')
    

## An example for computing margin bounds.
# In compute_bounds() function you can pass in a specification matrix C, which is a final linear matrix applied to the last layer NN output.
# For example, if you are interested in the margin between the groundtruth class and another class, you can use C to specify the margin.
# This generally yields tighter bounds.
# Here we compute the margin between groundtruth class and groundtruth class + 1.
# If you have more than 1 specifications per batch element, you can expand the second dimension of C (it is 1 here for demonstration).
lirpa_model = BoundedModule(model, torch.empty_like(my_input), device=my_input.device)
C = torch.zeros(size=(N, 1, n_classes), device=my_input.device)
groundtruth = my_input_label.to(device=my_input.device).unsqueeze(1)
print("groundtruth -->", groundtruth)
print(groundtruth.shape)
target_label = (groundtruth + 1) % n_classes
C.scatter_(dim=2, index=groundtruth, value=1.0)
C.scatter_(dim=2, index=target_label, value=-1.0)
print('Demonstration 3: Computing bounds with a specification matrix.\n')
print('Specification matrix:\n', C)

for method in ['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)', 'CROWN-Optimized (alpha-CROWN)']:
    print('Bounding method:', method)
    if 'Optimized' in method:
        # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
        lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1, }})
    lb, ub = lirpa_model.compute_bounds(x=(my_input,), method=method.split()[0], C=C)
    for i in range(N):
        print('input {} top-1 prediction {} ground-truth {}'.format(i, label[i], my_input_label[i].item()))
        print('margin bounds: {l:8.3f} <= f_{j}(x_0+delta) - f_{target}(x_0+delta) <= {u:8.3f}'.format(
            j=my_input_label[i], target=(my_input_label[i] + 1) % n_classes, l=lb[i][0].item(), u=ub[i][0].item()))
    print()

