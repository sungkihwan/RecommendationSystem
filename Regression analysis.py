# 라이브러리 호출
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv('./mba_admission.csv')

data.head()

class dataset(torch.utils.data.Dataset):
  def __init__(self,data):
    self.data = data
    self.data['gmat'] /= self.data['gmat'].max()
    self.data['gpa'] /= self.data['gpa'].max()
    self.data['work_experience'] /= self.data['work_experience'].max()

  def __getitem__(self,idx):
    X = torch.Tensor(self.data[self.data.columns[:3]].values)
    Y = torch.Tensor(self.data['admitted'])
    return X[idx,:], Y[idx]

  def __len__(self):
    return self.data.shape[0]

train_dataset = dataset(data)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=8,shuffle=True,drop_last=False)


# neural network 모델 생성
# 구조:  입력(3), 은닉(10), 출력(1)
# 활성화:         tanh      sigmoid
class admission_model(nn.Module):
    def __init__(self):
        super(admission_model, self).__init__()
        self.lin1 = nn.Linear(3, 10)
        self.lin2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.tanh(self.lin1(x))
        x = F.sigmoid(self.lin2(x))
        return x

model = admission_model()
optimizer = optim.SGD(model.parameters(),lr=0.5)

# 학습
for ep in range(100):
  loss_buffer = []
  for X,Y in train_loader:
    optimizer.zero_grad()
    y_infer = model(X).view(-1)
    loss = -torch.mean(Y*torch.log(y_infer)+(1-Y)*torch.log(1-y_infer)) # cross entropy 함수
    loss.backward()
    optimizer.step()
    loss_buffer.append(loss.item())

  if ep % 10 == 0:
    print('Epoch: {}, Loss: {}'.format(ep,np.mean(loss_buffer)))

# 결과 시각화
y_infer = model(torch.Tensor(data[data.columns[:3]].values))

plt.scatter(np.arange(data.shape[0]),data['admitted'],color='red',label='True result')
plt.scatter(np.arange(data.shape[0]),y_infer.detach().numpy(),color='blue',label='Predicted result')
plt.hlines(0.5,-10,60,label='Admission boundary')
plt.legend(loc='upper center',fontsize=' 10')
plt.xlim(-1,41)
plt.ylim(-0.1,1.5)
plt.xlabel('Volunteer')
plt.ylabel('Result')
plt.title('MBA admission result included prediction')