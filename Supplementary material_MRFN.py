import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 1.Data
data_file = 'data.csv'
output_dir = Path('mrfn_results')
data = pd.read_csv(data_file)

X = data.iloc[:, :79].to_numpy(dtype=np.float64)
y = data.iloc[:, 79:].to_numpy(dtype=np.float64)

# 2. Training, validation and test split (6:2:2)
train_idx, temp_idx = train_test_split(np.arange(len(data)), test_size=0.4, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
print(f'Training samples: {len(train_idx)}, validation samples: {len(val_idx)}, test samples: {len(test_idx)}')

# 3. Z-score standardisation fitted on the training set
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaler.fit(X[train_idx])
y_scaler.fit(y[train_idx])
X_scaled = X_scaler.transform(X)
y_scaled = y_scaler.transform(y)

# 4. Tensor shapes: [N, 1, 79] for inputs and [N, 18] for targets
X_data_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
y_data_tensor = torch.tensor(y_scaled, dtype=torch.float32)

X_train_tensor = X_data_tensor[train_idx]
X_val_tensor = X_data_tensor[val_idx]
y_train_tensor = y_data_tensor[train_idx]
y_val_tensor = y_data_tensor[val_idx]

# 5. Random seed and device
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_tensor = X_train_tensor.to(device)
X_val_tensor = X_val_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_val_tensor = y_val_tensor.to(device)

# 6. MRFN architecture
class MultiKernelCNN(nn.Module):
    def __init__(self):
        super(MultiKernelCNN, self).__init__()

        # Branch 1: kernel length 1
        self.branch1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Branch 2: kernel length 3
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Branch 3: kernel length 5
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 3 branches × 64 channels × 19 depth positions = 3648 features.
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 19, 256),
            nn.ReLU(),
            nn.Linear(256, 18)
        )

    def forward(self, x):
        b1 = self.branch1(x)  # [batch, 64, 19]
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        combined = torch.cat([b1, b2, b3], dim=1)
        flattened = combined.view(combined.size(0), -1)
        output = self.fc(flattened)
        return output


model = MultiKernelCNN().to(device)

# 7. MAE on standardised targets and Adam optimiser
mae_loss = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 8. Training and model selection by validation MAE
num_epochs = 100
batch_size = 32
best_val_mae = float('inf')
best_epoch = None
best_weights = None

for epoch in range(num_epochs):
    model.train()
    order = torch.randperm(len(X_train_tensor), device=device)
    train_error_sum = 0.0

    for i in range(0, len(X_train_tensor), batch_size):
        batch_idx = order[i:i + batch_size]
        X_batch = X_train_tensor[batch_idx]
        y_batch = y_train_tensor[batch_idx]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = mae_loss(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_error_sum += loss.item() * len(X_batch)

    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_tensor)
        val_mae = mae_loss(y_val_pred, y_val_tensor).item()

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_epoch = epoch + 1
        best_weights = deepcopy(model.state_dict())

    train_mae = train_error_sum / len(X_train_tensor)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train MAE: {train_mae:.6f}, Val MAE: {val_mae:.6f}')

if best_weights is None:
    raise RuntimeError('Training did not yield a finite validation MAE.')

model.load_state_dict(best_weights)
print(f'Best epoch: {best_epoch}, validation MAE: {best_val_mae:.6f}')

# 9. Predictions and test errors in the original units
model.eval()
pred_batches = []
with torch.no_grad():
    for i in range(0, len(X_data_tensor), batch_size):
        X_batch = X_data_tensor[i:i + batch_size].to(device)
        pred_batches.append(model(X_batch).cpu().numpy())

y_pred = y_scaler.inverse_transform(np.concatenate(pred_batches).astype(np.float64))
output_dir.mkdir(parents=True, exist_ok=True)
output_columns = list(data.columns[79:])

# Export predictions for all splits in the original row order.
split = np.empty(len(data), dtype=object)
split[train_idx], split[val_idx], split[test_idx] = 'train', 'validation', 'test'
prediction_table = data.copy()
prediction_table.insert(0, 'split', split)
predicted_columns = pd.DataFrame(y_pred, columns=[f'pred_{name}' for name in output_columns])
pd.concat([prediction_table, predicted_columns], axis=1).to_csv(
    output_dir / 'predictions.csv', index=False
)

# MAE and RMSE for each soil parameter on the held-out test set.
test_metrics = []
for i, name in enumerate(output_columns):
    error = y_pred[test_idx, i] - y[test_idx, i]
    mae_dim = np.mean(np.abs(error))
    rmse_dim = np.sqrt(np.mean(error ** 2))
    print(f'{name} - MAE: {mae_dim:.6f}, RMSE: {rmse_dim:.6f}')
    test_metrics.append((name, mae_dim, rmse_dim))

pd.DataFrame(test_metrics, columns=['parameter', 'mae', 'rmse']).to_csv(
    output_dir / 'test_metrics.csv', index=False
)

# 10. Save the selected weights and standardisation parameters
torch.save(
    {
        'model_state_dict': model.state_dict(),
        'input_columns': list(data.columns[:79]),
        'output_columns': output_columns,
        'input_mean': X_scaler.mean_.tolist(),
        'input_scale': X_scaler.scale_.tolist(),
        'output_mean': y_scaler.mean_.tolist(),
        'output_scale': y_scaler.scale_.tolist(),
        'best_epoch': best_epoch,
        'validation_mae_normalised': best_val_mae,
    },
    output_dir / 'mrfn.pth'
)

# Stage 3 measured wall deflections
input_columns = list(data.columns[:79])
output_columns = list(data.columns[79:])

field_data = pd.read_csv("measured.csv")

# Preserve training feature order
X_field = field_data[input_columns].to_numpy(dtype=np.float64)
X_field_scaled = X_scaler.transform(X_field)

X_field_tensor = (
    torch.tensor(X_field_scaled, dtype=torch.float32)
    .unsqueeze(1)
    .to(device)
)

# Soil parameter inversion
model.eval()
with torch.no_grad():
    y_field_scaled = model(X_field_tensor).cpu().numpy()

# Restore physical units
soil_parameters = y_scaler.inverse_transform(y_field_scaled)

pd.DataFrame(
    soil_parameters,
    columns=output_columns
).to_csv(output_dir / "inversed_soil_parameters.csv", index=False)
