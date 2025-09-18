import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 전처리
try:
    # 'data.csv' 파일을 pandas를 이용해 불러옵니다.
    df = pd.read_csv('data/hourly_stats.csv')
    print("CSV 파일 로드 완료. 데이터 정보:")
    print(df.head())
except FileNotFoundError:
    print("Error: 'data.csv' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    exit()

# '_time' 컬럼을 x축 데이터로 사용하기 위해 날짜/시간 형식으로 변환합니다.
# format 인자를 제거하여 pandas가 자동으로 날짜 형식을 감지하도록 합니다.
df['_time'] = pd.to_datetime(df['_time'], errors='coerce')
# 'errors=coerce'는 변환 실패 시 NaT(Not a Time)로 만듭니다.
df = df.dropna(subset=['_time']) # 변환 실패한 행 제거

# 데이터가 비어있는지 확인
if df.empty:
    print("Error: 날짜/시간 형식 변환 후 데이터가 모두 제거되었습니다. '_time' 컬럼의 형식을 확인해주세요.")
    exit()

# 'x' 컬럼을 시계열 데이터로 사용합니다.
data_series = df['x'].values.reshape(-1, 1)

# 데이터 정규화 (Normalization)
# LSTM 모델의 학습 효율을 높이기 위해 데이터를 0과 1 사이로 정규화합니다.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_series)

# 시퀀스 데이터셋 생성
# 과거 데이터를 'look-back' 기간만큼 묶어서 입력(X)으로, 다음 값을 정답(y)으로 설정합니다.
sequence_length = 5  # 과거 5개의 데이터로 다음 값 예측
X = []
y = []
for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i:i + sequence_length])
    y.append(scaled_data[i + sequence_length])

X = np.array(X)
y = np.array(y)

# 훈련 및 테스트 데이터 분리 (80% 훈련, 20% 테스트)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Pytorch 텐서로 변환
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

# 2. LSTM 모델 구축
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. 모델 학습
epochs = 200
for epoch in range(epochs):
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 4. 예측 및 시각화
model.eval()
with torch.no_grad():
    # 전체 데이터셋을 사용한 예측
    full_data_predictions = model(torch.FloatTensor(X))

# 정규화된 값을 원래 스케일로 되돌리기
# .detach()를 추가하여 그래디언트 계산 그래프에서 분리
full_predictions_original = scaler.inverse_transform(full_data_predictions.detach().numpy())
test_predictions_original = scaler.inverse_transform(model(X_test_t).detach().numpy())

# 시각화를 위한 전체 데이터 및 예측값 준비
combined_predictions = np.full_like(data_series, np.nan)
combined_predictions[sequence_length:] = full_predictions_original

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(df['_time'], data_series, label='Real data', color='blue')
plt.plot(df['_time'][sequence_length:], combined_predictions[sequence_length:], label='Pred. Data', color='red', linestyle='--')
plt.title('LSTM results')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 마지막 값 예측
last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
last_sequence_t = torch.FloatTensor(last_sequence)
with torch.no_grad():
    future_prediction_scaled = model(last_sequence_t)

future_prediction_original = scaler.inverse_transform(future_prediction_scaled.numpy())
print(f"다음 날 예측 값: {future_prediction_original[0][0]:.4f}")
