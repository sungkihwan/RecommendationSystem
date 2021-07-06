import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Cuda
print(torch.cuda.is_available())

print(torch.cuda.current_device())

print(torch.cuda.device_count())

print(torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Model Save
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.lin1 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.lin1(x)
        return x

test_model = model().to(device)
optimizer = optim.SGD(test_model.parameters(),lr=0.1)

x_data = np.linspace(-1,1,100)
y_data = np.sin(x_data)+np.random.uniform(-1,1,size=100)
x = torch.Tensor(x_data).view(-1,1).to(device)
y = torch.Tensor(y_data).view(-1,1).to(device)

for ep in range(10):
  optimizer.zero_grad()
  y_infer = test_model(x)
  loss = torch.mean((y_infer-y)**2)
  loss.backward()
  optimizer.step()
  print(loss.item())

# 현재까지 학습이 다 완료된 상태 세이브
torch.save(test_model,'./model_save.pth')

del test_model

# 모델 로딩 -> 주의 저장된 모델에 대한 소스코드의 정보는 앞서서 정의가 되어있어야 한다.
test_model = torch.load('./model_save.pth')

# Tensor -> numpy
x = torch.Tensor(torch.linspace(-1,1)).view(-1,1).to(device)

y_infer = test_model(x)
print(y_infer.mean())

y_numpy = y_infer.detach().cpu().numpy() # Graph 연결해제, Cuda 해제, Tensor -> Numpy 변환
print(y_numpy.mean())