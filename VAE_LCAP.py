import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
data = pd.read_csv('output_with_return.csv')
X_vars = [
    'earnings_to_price_ratio', 'book_to_price_ratio', 'cash_earnings_to_price_ratio',
    'sales_to_price_ratio', 'admin_expense_rate', 'book_leverage',
    'cash_to_current_liability', 'current_ratio', 'eps_ttm',
    'equity_to_fixed_asset_ratio', 'fixed_asset_ratio', 'gross_income_ratio',
    'intangible_asset_ratio', 'market_leverage', 'operating_cost_to_operating_revenue_ratio',
    'quick_ratio', 'roa_ttm', 'roe_ttm', 'net_invest_cash_flow_ttm',
    'net_asset_growth_rate', 'net_profit_growth_rate', 'operating_profit_growth_rate',
    'operating_revenue_growth_rate', 'net_operate_cashflow_growth_rate',
    'total_asset_growth_rate', 'total_profit_growth_rate'
]
data.dropna(subset=X_vars + ['natural_log_of_market_cap', 'return'], inplace=True)
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
data[X_vars] = scaler_X.fit_transform(data[X_vars])

scaler_T = StandardScaler()
data['natural_log_of_market_cap'] = scaler_T.fit_transform(data[['natural_log_of_market_cap']])

scaler_Y = StandardScaler()
data['return'] = scaler_Y.fit_transform(data[['return']])

# Clip values to prevent extreme outliers
data[X_vars] = data[X_vars].clip(-5, 5)
data['natural_log_of_market_cap'] = data['natural_log_of_market_cap'].clip(-5, 5)
data['return'] = data['return'].clip(-5, 5)

class FinanceDataset(Dataset):
    def __init__(self, df):
        self.X = df[X_vars].values.astype(np.float32)
        self.T = df['natural_log_of_market_cap'].values.astype(np.float32).reshape(-1, 1)
        self.Y = df['return'].values.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.Y[idx]

dataset = FinanceDataset(data)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2_mu = nn.Linear(128, latent_dim)
        self.fc2_logvar = nn.Linear(128, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)
        # Predictor
        self.fc5 = nn.Linear(latent_dim + 1, 64)
        self.fc6 = nn.Linear(64, 1)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        mu = self.fc2_mu(h1)
        logvar = self.fc2_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return self.fc4(h3)

    def predict(self, z, t):
        zt = torch.cat([z, t], dim=1)
        h4 = torch.relu(self.fc5(zt))
        return self.fc6(h4)

    def forward(self, x, t):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        y_pred = self.predict(z, t)
        return x_recon, mu, logvar, y_pred
def loss_function(x_recon, x, mu, logvar, y_pred, y_true, alpha=1.0, beta=0.1, gamma=1.0):
    # Reconstruction loss
    recon_loss = nn.MSELoss()(x_recon, x)
    # KL divergence loss (weighted)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # Prediction loss
    pred_loss = nn.MSELoss()(y_pred, y_true)
    # Total loss with weighting factors
    total_loss = alpha * recon_loss + beta * kl_loss + gamma * pred_loss
    return total_loss, recon_loss.item(), kl_loss.item(), pred_loss.item()

# Define the model and apply weight initialization
input_dim = len(X_vars)
latent_dim = 3  # You can adjust this

model = VAE(input_dim, latent_dim)

# Initialize weights
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(weights_init)

# Optimizer with reduced learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-2)

model.train()
num_epochs = 15
for epoch in range(num_epochs):
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_pred_loss = 0
    for x_batch, t_batch, y_batch in dataloader:
        optimizer.zero_grad()
        x_recon, mu, logvar, y_pred = model(x_batch, t_batch)
        loss, recon_loss_value, kl_loss_value, pred_loss_value = loss_function(
            x_recon, x_batch, mu, logvar, y_pred, y_batch,
            alpha=1.0, beta=0.1, gamma=1.0
        )
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_recon_loss += recon_loss_value * batch_size
        total_kl_loss += kl_loss_value * batch_size
        total_pred_loss += pred_loss_value * batch_size
    avg_loss = total_loss / len(dataset)
    avg_recon_loss = total_recon_loss / len(dataset)
    avg_kl_loss = total_kl_loss / len(dataset)
    avg_pred_loss = total_pred_loss / len(dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, '
          f'KL Loss: {avg_kl_loss:.4f}, Pred Loss: {avg_pred_loss:.4f}')

model.eval()
with torch.no_grad():
    X_tensor = torch.tensor(dataset.X)
    T_tensor = torch.tensor(dataset.T)
    Y_true = torch.tensor(dataset.Y)

    # Encode to get latent Z
    mu, logvar = model.encode(X_tensor)
    Z = model.reparameterize(mu, logvar)

    # Predict Y with T (actual treatment)
    Y_pred_actual = model.predict(Z, T_tensor)

    # Predict Y with T set to zero (control)
    T_zero = torch.zeros_like(T_tensor)
    Y_pred_control = model.predict(Z, T_zero)

    # Calculate ITE
    ITE = (Y_pred_actual - Y_pred_control).numpy()
from scipy import stats

# Flatten arrays
ite_values = ITE.flatten()
y_true_values = Y_true.numpy().flatten()

# Calculate T-statistic and P-value
t_statistic, p_value = stats.ttest_1samp(ite_values, 0)

# Output ITE values
print("First 10 ITE values:")
print(ite_values[:10])
average_ite = sum(ite_values) / len(ite_values)
max_ite = max(ite_values)

print(f"ITE 平均值: {average_ite}")
print(f"ITE 最大值: {max_ite}")
# Output T-statistic and P-value
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")
