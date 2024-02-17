# auto_LiRPA 간단 사용 가이드
---------------------------------
> 먼저, 제가 작업한 코드는 오직 model.py뿐이고 나머지는 전부 auto_LiRPA 툴을 사용하기 위한 코드이므로 교수님께서는 model.py만 원하시는 대로 조작할 것을 권장드립니다.
> 원본 README.md는 README_original.md로 보존해두었습니다.

## 가이드
```python
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

# Define computation as a nn.Module.
class MyModel(nn.Module):
    def forward(self, x):
        # Define your computation here.

model = MyModel()
my_input = load_a_batch_of_data()
# Wrap the model with auto_LiRPA.
model = BoundedModule(model, my_input)
# Define perturbation. Here we add Linf perturbation to input data.
ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
# Make the input a BoundedTensor with the pre-defined perturbation.
my_input = BoundedTensor(my_input, ptb)
# Regular forward propagation using BoundedTensor works as usual.
prediction = model(my_input)
# Compute LiRPA bounds using the backward mode bound propagation (CROWN).
lb, ub = model.compute_bounds(x=(my_input,), method="backward")
```
1. 먼저 pytorch 기반으로 생성한 모델과 tensor기반의 input이 필요합니다.
2. 해당 모델을 BoundedModule(model, my_input)을 통해 감싸줍니다. 아마 model을 abstraction하는 과정으로 추측됩니다.
3. PerturbationNorm(norm=np.inf, eps=0.1)은 검사할 기준을 뜻합니다. 교수님께서 제공해주신 자료의 norm값을 설정할 수 있으며 교재에도 나와 있듯 norm=np.inf는 전체적인 perturbation에 대응하는지, norm=2는 극히 일부분의 input의 큰 변화에 대응하는지에 대한 검사이며 eps는 그 변화량을 뜻합니다.
4. BoundedTensor(my_input, ptb)로 input에 perturbation을 적용시킵니다.
5. 이제 model.compute_bounds를 통해 각 input에 대하여 lowerbound와 upperbound를 값으로 받을 수 있습니다. 이 xor 신경망의 경우 method를 별도로 지정하지 않아도 됩니다.
## 주의사항
+ 반드시 출력에 해당하는 label이 있어야 합니다. xor의 경우에는 0(false) 1(true)의 2개의 label을 가지므로 총 2개의 출력노드를 가져야 합니다.
