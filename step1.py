import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
data = pd.read_csv('selected_variables.csv')
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()
numeric_cols = data.select_dtypes(include=[np.number]).columns
data = data[np.isfinite(data[numeric_cols]).all(axis=1)]
features = [
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
data['mispricing'] = np.nan
data['date'] = pd.to_datetime(data['date'])
time_groups = data.groupby('date')
for date, group in time_groups:
    print(date)
    group = group.sort_values('stkcd')
    X = group[features].values
    y = group['natural_log_of_market_cap'].values
    print(len(X),len(y))
    '''if len(X) < 10:  
        continue'''
    model = AdaBoostRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    actual_market_cap = np.exp(y)
    predicted_market_cap = np.exp(y_pred)
    mispricing = (predicted_market_cap - actual_market_cap) / actual_market_cap
    data.loc[group.index, 'mispricing'] = mispricing
output_file = 'output_with_mispricing.csv'
data.to_csv(output_file, index=False)
print(f"错误定价变量已计算并保存至 {output_file}")