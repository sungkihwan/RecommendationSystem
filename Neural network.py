# PyTorch 필요 라이브러리 호출

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Neural network 모델 생성
# 입력 -> 은닉 (3,5)
# 은닉 -> 출력 (5,2)
class fir_model(nn.Module):
    def __init__(self):
        super(fir_model, self).__init__()
        # 가중치 행렬 1
        self.lin1 = nn.Linear(3, 5)
        # 가중치 행렬 2
        self.lin2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.lin1(x)  # 3,5 행렬 통과
        x = F.relu(x)  # relu 함수 통과
        x = self.lin2(x)  # 5,2 행렬 통과
        x = F.sigmoid(x)  # sigmoid 함수 통과
        return x

# Neural network 모델 정의 및 최적화 툴(optimizer 사용예시)
model = fir_model()
opt = optim.SGD(model.parameters(), lr=0.01) # Adam, RMSprop -> SGD

# Backward propagation
# 최적화 단계 이 셀은 올바르게 실행되지 않아요! x,y는 실제론 없지만 있다고 가정하고 코딩
criterion = nn.MSELoss() # criterion 변수는 (MSELoss, (y_-y)**2)

x = torch.Tensor(x) # 입력 데이터(3차원벡터)
y = torch.Tensor(y) # 출력 데이터(2차원벡터)

opt.zero_grad() #optimizer 안 모든 gradient 초기화
y_infer = model(x)  # Foward propagation
loss = criterion(y_infer, y)
loss.backward()
opt.step()