#!/usr/bin/env python
# coding: utf-8

# # 모델 준비

# ## Time layer

# ### TimeLSTM

# In[1]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LSTM:
    def __init__(self, Wx, Wh, b):
        '''
        Parameters
        ----------
        Wx: 입력 x에 대한 가중치 매개변수(4개분의 가중치가 담겨 있음)
        Wh: 은닉 상태 h에 대한 가장추 매개변수(4개분의 가중치가 담겨 있음)
        b: 편향（4개분의 편향이 담겨 있음）
        '''
        self.params = [Wx, Wh, b] #W, Wh ,b를 params에 할당
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]  #이에 대응하는 형태로 기울기도 초기화
        self.cache = None #cache는 순전파때 중간 결과를 보관했다가 역전파 계산에 사용하려는 용도의 인스턴스 변수
        
        '''
        x: 현 시각의 입력데이터
        h_prev: 이전 시각의 은닉 상태
        c_prev: 이전 시각의 기억 셀
        
        '''
    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params 
        N, H = h_prev.shape #N: 미니배치수, H: 은닉 상태의 차원수 

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b #네 개분의 아핀 변환 결과를 저장

        f = A[:, :H] 
        g = A[:, H:2*H] 
        i = A[:, 2*H:3*H] 
        o = A[:, 3*H:] 
        '''
        위와 같이 슬라이스해서 꺼내고 이를 연산 노드에 분배
        '''
        f = sigmoid(f) #forget gate
        g = np.tanh(g) #기억셀
        i = sigmoid(i) #input gate
        o = sigmoid(o) #output gate

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)

        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)

        dc_prev = ds * f

        di = ds * g 
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i

        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g ** 2)

        dA = np.hstack((df, dg, di, do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False): #stateful가 True인 경우 은닉 상태를 유지하고, False인 경우 은닉 상태를 초기화
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None #LSTM 계층을 리스트로 저장하는 용도

        self.h, self.c = None, None #h: forward 메서드를 불러왔을때의 마지막 LSTM 계층의 은닉 상태를 저정
        self.dh = None #dh: backward 메서드를 불러왔을때 앞 블록의 은닉 상태 기울기를 저장
        self.stateful = stateful

    def forward(self, xs): #xs: T개 분량의 시계열 데이터를 하나로 모은것
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype='f') #출력값을 저장할 그릇 생성

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T): #t번 반복해서 생성된 LSTM 계층을 생성하고 변수를 layers에 추가한다.
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None


# ### timeaffine

# In[2]:


class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N*T, -1)
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N*T, -1)
        rx = x.reshape(N*T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


# ### Timemse

# In[3]:


class MSE:
    def __init__(self):
        self.x = None
        self.y = None
        self.loss = None

    def forward(self, x,y):
        self.x = x
        self.y = y
        self.loss = np.square(x-y).mean()
        return self.loss

    def backward(self,dout):
        dout=dout*(self.x-self.y)
        return dout




class TimeMSE:
    def __init__(self):
        self.params, self.grads = [], []
        self.xs_shape = None
        self.layers = None

    def forward(self, xs, ts):
        xs1=xs.squeeze()
        ts1=ts.squeeze()
        N, T = xs1.shape
        self.xs_shape = xs1.shape

        self.layers = []
        loss = 0

        for t in range(T):
            layer = MSE()
            loss += layer.forward(xs1[:, t], ts1[:, t])
            self.layers.append(layer)

        
        return loss / T

    def backward(self, dout=1):
        N, T = self.xs_shape
        dxs = np.empty(self.xs_shape, dtype='f')

        dout *= 1/T
        for t in range(T):
            layer = self.layers[t]
            dxs[:, t] = layer.backward(dout)

        return dxs


# ## learning model

# ### Rnnfc

# In[4]:


import pickle
import numpy as np
class Rnnfc:
    def __init__(self, dv_size=100, hidden_size=100):
        D, H = dv_size, hidden_size
        rn = np.random.randn

        # 가중치 초기화, Xavier 초깃값 이용
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, 1) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(1).astype('f')

        # 계층 생성
        self.layers = [
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeMSE()
        self.lstm_layer = self.layers[0]

        # 모든 가중치와 기울기를 리스트에 모은다.
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.lstm_layer.reset_state()
        
    def save_params(self, file_name='Rnnfc.pkl'):
        with open(file_name, 'wb') as f:
            self.dump(self.params,f)
    
    def load_params(self, file_name='Rnnfc.pkl'):
        with open(file_name, 'rb') as f:
            self.params=pickle.load(f)       


# ## optimizer

# ### ADAM

# In[5]:


class Adam:
    '''
    Adam (http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)


# # 데이터 전처리

# In[6]:


import pandas as pd
df=pd.read_csv("just/temp1.csv",encoding="euc-kr")
df=df.iloc[:,3:]
df.fillna(method='ffill',inplace=True)


# In[7]:


df.columns


# In[8]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# 스케일을 적용할 column을 정의합니다.
scale_cols = ['평균 이슬점온도(°C)', '평균 증기압(hPa)', '평균 현지기압(hPa)',
       '평균 해면기압(hPa)', '가조시간(hr)', '평균 지면온도(°C)', '최저 초상온도(°C)',
       '평균 5cm 지중온도(°C)']
# 스케일 후 columns
df_scaled = scaler.fit_transform(df[scale_cols])
df_scaled


# In[9]:


df_s = pd.DataFrame(df_scaled, columns=scale_cols)


# In[10]:


df_s


# In[11]:

#train 데이터, test 데이터 구분하기
x_train=df_s[:2900] 
x_test=df_s[2900:]
y_train=df["평균기온(°C)"][:2900]
y_test=df["평균기온(°C)"][2900:]


# In[12]:


x_train.shape, y_train.shape


# In[13]:


x_test.shape, y_test.shape


# In[14]:


x_train[0:1]


# # 모델 테스트

# ## 훈련

# In[15]:


# 하이퍼파라미터 설정
batch_size = 16
dv_size = 8
hidden_size = 8 # RNN의 은닉 상태 벡터의 원소 수
time_size = 10   # Truncated BPTT가 한 번에 펼치는 시간 크기
lr = 0.05
max_epoch = 500
#max_grad = 0.25

xs=x_train[:-1][:]
ts=y_train[1:] #다음날 온도를 측정하고 싶기 때문이다.
data_size=len(x_train)
# 모델 생성
model = Rnnfc(dv_size,hidden_size)
optimizer = Adam(lr)


# In[16]:


max_iters = data_size // (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0
loss_list = []
jump = (data_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]


# In[17]:


print(jump,offsets)


# In[18]:


for epoch in range(max_epoch):
    for iter in range(max_iters):
        # 미니배치 취득
        batch_x = np.empty((batch_size, time_size,8), dtype='f')
        batch_t = np.empty((batch_size, time_size,1), dtype='f')
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size:((offset + time_idx) % data_size)+1]
                batch_t[i, t] = ts[((offset + time_idx) % data_size)+1]
            time_idx += 1

        # 기울기를 구하여 매개변수 갱신
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1

    # 에폭마다 로스 평가
    print('| 에폭 %d | loss %.2f'% (epoch+1, total_loss))
    loss_list.append(float(total_loss))
    total_loss, loss_count , time_idx= 0, 0, 0


# In[19]:


import matplotlib.pyplot as plt
'''
x = np.arange(len(loss_list))
plt.plot(x, loss_list, label='train')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
'''

# ## 테스트

# In[20]:


batch_size=16
time_size=1
xs=x_test[:-1][:]
ts=y_test[1:]
data_size=len(x_test)
max_iters = data_size // (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0
loss_list = []
jump = (data_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]
predict=[]


# In[21]:


for iter in range(max_iters):
        # 미니배치 취득
        batch_x = np.empty((batch_size, time_size,8), dtype='f')
        batch_t = np.empty((batch_size, time_size,1), dtype='f')
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs.iloc[(offset + time_idx) % data_size:((offset + time_idx) % data_size)+1]
            time_idx += 1

        # 기울기를 구하여 매개변수 갱신
        predict.append(model.predict(batch_x))


# In[24]:


x_pre=[]
for i in range(batch_size):
    for j in range(max_iters):
        x_pre.append(predict[j][i][0][0])
ts=ts.reset_index()["평균기온(°C)"] [:720]


# In[28]:


x=range(len(x_pre))
plt.figure(figsize=(20,8))
plt.plot(x, x_pre, label ='predict')
plt.plot(x, ts, label ='temperature')
plt.legend()
plt.show()
