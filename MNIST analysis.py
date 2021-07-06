# 라이브러리 호출
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dataset
import torchvision.transforms as transforms # 텐서로 변환

import numpy as np
import matplotlib.pyplot as plt

# 데이터 다운로드
mnist_train = dataset.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dataset.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

print(mnist_train)
print(mnist_test)
# sample
img, label = mnist_train[0]
plt.imshow(img[0,:,:])
print(label)

# 상대적으로 쉬운 문제이므로 valid_dataset은 하지 않음
train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=32,shuffle=True,drop_last=False)
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=32,shuffle=False,drop_last=False)

# Neural network 모델만들기
class MNIST_full(nn.Module): # fully connected -> 784 - 256 연결되어있음
  def __init__(self):
    super(MNIST_full,self).__init__()
    self.lin1 = nn.Linear(784,256) # 입력 -> 은닉1
    self.lin2 = nn.Linear(256,128) # 은닉1 -> 은닉2
    self.lin3 = nn.Linear(128,10) # 은닉2 -> 출력

  def forward(self,x):
    x = x.view(-1,784) # (batch, 28, 28, 1) -> (batch, 784)
    x = F.relu(self.lin1(x))
    x = F.relu(self.lin2(x))
    x = F.softmax(self.lin3(x),dim=1)
    return x

model = MNIST_full()
optimizer = optim.Adam(model.parameters(),lr=0.0001) # Adam은 SGD의 업그레이드 , lr = 0.0001

def kl_div(prob1, prob2):
  return torch.sum(prob1*torch.log(prob1/prob2+1e-15)) # 1e-15 를 더해줌 (log0 방지)

def cross(prob1,prob2):
  return -torch.sum(prob1*torch.log(prob2))

img, label = mnist_train[0]
y_onehot = torch.zeros(10)
y_onehot[label]=1
y_onehot = y_onehot.view(1,-1)
y_infer = model(img)
print(y_infer)
print(cross(y_onehot,y_infer))

# Train 단계
criterion = nn.CrossEntropyLoss()  # ki_div 보다 좀 더 가벼운 확률간 거리를 구하는 유틸
for epoch in range(10):
    loss_buffer = []
    for x, y in train_loader:  # x : 이미지 , y : 레이블
        optimizer.zero_grad()  # gradient 0 초기화

        y_onehot = torch.zeros((y.shape[0], 10))
        y_onehot[range(y.shape[0]), y] = 1  # one_hot encoding 3 -> [0,0,0,1,0,0,0,0,0,0]
        y_infer = model(x)  # foward propagation
        # loss = criterion(y_infer,y)
        loss = kl_div(y_onehot, y_infer)  # ki_div 확률간 거리 구하기(레이블,, 추론)
        loss.backward()  # gradient 획득
        optimizer.step()  # SGD 수행
        loss_buffer.append(loss.item())

    print('Epoch: {:d}. Train loss {:f}'.format(epoch, np.mean(loss_buffer)))

# Test 단계
acc = []
for x, y in test_loader:
    # one_hot encoding
    y_onehot = torch.zeros((len(y), 10))
    y_onehot[range(len(y)), y] = 1
    y_infer = model(x)  # foward propagation

    # 정확도 계산
    correct_prediction = torch.argmax(y_infer, 1) == y
    acc.append(correct_prediction.float().mean())

print('Test loss {:f}, Accuracy {:f}'.format(np.mean(loss_buffer), np.mean(acc)))

# 실물샘플
img,label=mnist_test[0]
plt.imshow(img[0,:,:])
y_infer = model(img)
print('실제 이미지의 숫자',label)
print('예측 이미지의 숫자',torch.argmax(y_infer,1))