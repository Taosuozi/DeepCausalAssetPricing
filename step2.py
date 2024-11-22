import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('output_with_mispricing.csv')
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')

x = 24
c = 20
results = []

grouped = data.groupby('date')
dates = sorted(data['date'].unique())

for i in range(len(dates) - x):
    current_date = dates[i]
    print(current_date)
    date_x = dates[i + x]

    current_group = grouped.get_group(current_date)
    group_x = grouped.get_group(date_x).set_index('stkcd')

    current_group = current_group.sort_values('mispricing')

    n = len(current_group) // c
    q1 = current_group.head(n)
    q_all = current_group
    q5 = current_group.tail(n)
    def calculate_r(group):
        total_return = 0
        col = 0
        for _, row in group.iterrows():
            stkcd = row['stkcd']
            if stkcd in group_x.index:
                current_price = row['price_no_fq']
                future_price = group_x.loc[stkcd, 'price_no_fq']
                total_return += (future_price - current_price) / current_price
                col += 1
        avg_return = total_return / col

        '''if current_date.year == 2019 and current_date.month >= 12 or current_date.year == 2020 and current_date.month <= 11:
            avg_return = -avg_return'''
        return avg_return

    def calculate_all(group):
        total_return = 0
        col = 0
        for _, row in group.iterrows():
            stkcd = row['stkcd']
            if stkcd in group_x.index:
                current_price = row['price_no_fq']
                future_price = group_x.loc[stkcd, 'price_no_fq']
                r = (future_price - current_price) / current_price
                total_return += r
                col += 1
                '''idx = data[(data['stkcd'] == stkcd)].index[0] + i
                data.at[idx, 'return'] = r'''
        avg_return = total_return / col

        '''if current_date.year == 2019 and current_date.month >= 12 or current_date.year == 2020 and current_date.month <= 11:
            avg_return = -avg_return'''
        return avg_return

    r1 = calculate_r(q1)
    r5 = calculate_r(q5)

    avg_return = calculate_all(q_all)
    results.append({
        'date': current_date,
        'avg':avg_return,
        'r1_minus_avg': r1 - avg_return,
        'r5_minus_avg': r5 - avg_return,
        'r5':r5,
        'r1':r1,
        'r5-r1':r5-r1
    })

results_df = pd.DataFrame(results)


plt.figure(figsize=(12, 6))

plt.plot(results_df['date'], results_df['r5_minus_avg'], label='Q5', marker='o')
plt.plot(results_df['date'], results_df['r1_minus_avg'], label='Q1', marker='o')
#plt.plot(results_df['date'], results_df['avg'], label='Average Return', marker='o')

plt.axhline(0, color='gray', linestyle='--', linewidth=1)

plt.xlabel('Date')
plt.ylabel('Return Minus Average Return')
plt.title(f'Q5 and Q1 Returns Minus Average Return Over Time (x={x} months)')
plt.legend()
plt.grid()
plt.show()
results_df['r5_minus_r1'] = results_df['r5_minus_avg'] - results_df['r1_minus_avg']

plt.figure(figsize=(12, 6))
plt.plot(results_df['date'], results_df['r5_minus_r1'], label='Q5 - Q1 Return Difference', marker='o', color='orange')

plt.axhline(0, color='gray', linestyle='--', linewidth=1)

plt.xlabel('Date')
plt.ylabel('R5 - R1 Return Difference')
plt.title(f'Q5 - Q1 Return Difference Over Time (x={x} months)')
plt.legend()
plt.grid()
plt.show()
results_df['r5_minus_r1'] = results_df['r5_minus_avg'] - results_df['r1_minus_avg']
results_df['combined'] = results_df[['r5_minus_avg', 'r1_minus_avg']].stack().reset_index(drop=True)

std_combined = results_df['combined'].std()
mean_return = results_df['r5_minus_r1'].mean()
risk = std_combined
risk_free_rate = 0
sharpe_ratio = (mean_return - risk_free_rate) / risk if risk != 0 else 0

print("result:")
print(f"return: {mean_return:.4f}")
print(f"risk(std): {risk:.4f}")
print(f"sharp ratio: {sharpe_ratio:.4f}")
#data.to_csv('output_with_return.csv', index=False)