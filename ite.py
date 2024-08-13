import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LinearRegression
from scipy.stats import t

# 数据读取
data = pd.read_feather('jqbarraMonthly.feather')

# 提取相关变量
X = data[['Mispricing', 'LCAP']]  # 可观测变量
Y = data['returns']  # 目标变量，假设已经有超额收益
other_features = data.drop(columns=['Mispricing', 'LCAP', 'returns'])


# 数据集定义
class StockDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X.iloc[idx].values, self.Y.iloc[idx]


# VAE模型定义
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, z


# 训练VAE
latent_dim = 10
vae = VAE(input_dim=X.shape[1], latent_dim=latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# 数据加载
dataset = StockDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 训练过程
for epoch in range(50):
    for batch_X, _ in dataloader:
        recon_X, mu, logvar, z = vae(batch_X.float())
        recon_loss = nn.MSELoss()(recon_X, batch_X.float())
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 使用生成的 Z 进行因果推断
with torch.no_grad():
    _, _, _, Z = vae(torch.tensor(X.values).float())

# 使用线性回归进行因果效应分析
lr = LinearRegression()
lr.fit(Z, Y)
coef = lr.coef_
pred = lr.predict(Z)
residuals = Y - pred
t_stats = coef / (residuals.std() / (len(Y) ** 0.5))

# 计算 p 值
p_values = [2 * (1 - t.cdf(abs(t_stat), df=len(Y) - 1)) for t_stat in t_stats]

# 输出结果
results = pd.DataFrame({
    'Variable': ['Mispricing', 'LCAP'],
    'ITE': coef,
    'T-statistic': t_stats,
    'P-value': p_values
})

print(results)
