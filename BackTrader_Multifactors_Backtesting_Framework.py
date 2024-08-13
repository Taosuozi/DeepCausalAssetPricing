# 加载需要的库
import time
import math
import datetime
import numpy as np
import pandas as pd

import tushare as ts
import backtrader as bt
from sklearn.svm import SVR

from ast import literal_eval
import matplotlib.pyplot as plt
from backtrader.feeds import PandasData
from sklearn.preprocessing import RobustScaler

# 实例化支持向量回归（SVR）模型
svr = SVR()

# 获取已清洗好的全A股列表
stocklist_allA = pd.read_csv('Data/selected_variables.csv', usecols=['stkcd'])
stocklist_allA = stocklist_allA['stkcd'].drop_duplicates().tolist()

print(stocklist_allA)

# 获取已清洗好的全A股所有数据
df_all = pd.read_csv('Data/selected_variables.csv')
df_all.columns = df_all.columns.str.strip()  # 去除列名前后的空格
df_all.columns = df_all.columns.str.replace(r'\s+', '', regex=True)  # 去除列名中的所有空白字符
print(df_all['date'])
# del df_all['date']
df_all['date'] = pd.to_datetime(df_all['date'])

'''
获得某一股票的全部数据
输入：code--该股票对应的ts_code
输出：df_stock--该股票的全部数据，存为df
'''


def get_stock_data(code):
    df_stock = df_all[df_all['stkcd'] == code]
    if df_stock.empty:
        print(f"股票代码 {code} 对应的数据为空！")
    df_stock = df_stock[
        ['date', 'natural_log_of_market_cap',
         'earnings_to_price_ratio', 'book_to_price_ratio', 'cash_earnings_to_price_ratio', 'sales_to_price_ratio',
         'admin_expense_rate', 'book_leverage', 'cash_to_current_liability', 'current_ratio', 'eps_ttm',
         'equity_to_fixed_asset_ratio', 'fixed_asset_ratio', 'gross_income_ratio', 'intangible_asset_ratio',
         'market_leverage', 'operating_cost_to_operating_revenue_ratio', 'quick_ratio', 'roa_ttm', 'roe_ttm',
         'net_invest_cash_flow_ttm', 'net_asset_growth_rate', 'net_profit_growth_rate', 'operating_profit_growth_rate',
         'operating_revenue_growth_rate', 'net_operate_cashflow_growth_rate', 'total_asset_growth_rate',
         'total_profit_growth_rate']]
    df_stock.index = df_stock.date
    df_stock = df_stock.sort_index()

    if df_stock.empty:
        print(f"股票代码 {code} 在所选日期范围内无数据！")

    return df_stock


# 修改原数据加载模块，以便能够加载更多自定义的因子数据
class Addmoredata(PandasData):
    lines = ('natural_log_of_market_cap',
             'earnings_to_price_ratio', 'book_to_price_ratio', 'cash_earnings_to_price_ratio', 'sales_to_price_ratio',
             'admin_expense_rate', 'book_leverage', 'cash_to_current_liability', 'current_ratio', 'eps_ttm',
             'equity_to_fixed_asset_ratio', 'fixed_asset_ratio', 'gross_income_ratio', 'intangible_asset_ratio',
             'market_leverage', 'operating_cost_to_operating_revenue_ratio', 'quick_ratio', 'roa_ttm', 'roe_ttm',
             'net_invest_cash_flow_ttm', 'net_asset_growth_rate', 'net_profit_growth_rate',
             'operating_profit_growth_rate',
             'operating_revenue_growth_rate', 'net_operate_cashflow_growth_rate', 'total_asset_growth_rate',
             'total_profit_growth_rate',)
    params = (('date', 1), ('natural_log_of_market_cap', 3),
              ('earnings_to_price_ratio', 4),
              ('book_to_price_ratio', 5),
              ('cash_earnings_to_price_ratio', 6),
              ('sales_to_price_ratio', 7),
              ('admin_expense_rate', 8),
              ('book_leverage', 9),
              ('cash_to_current_liability', 10),
              ('current_ratio', 11),
              ('eps_ttm', 12),
              ('equity_to_fixed_asset_ratio', 13),
              ('fixed_asset_ratio', 14),
              ('gross_income_ratio', 15),
              ('intangible_asset_ratio', 16),
              ('market_leverage', 17),
              ('operating_cost_to_operating_revenue_ratio', 18),
              ('quick_ratio', 19),
              ('roa_ttm', 20),
              ('roe_ttm', 21),
              ('net_invest_cash_flow_ttm', 22),
              ('net_asset_growth_rate', 23),
              ('net_profit_growth_rate', 24),
              ('operating_profit_growth_rate', 25),
              ('operating_revenue_growth_rate', 26),
              ('net_operate_cashflow_growth_rate', 27),
              ('total_asset_growth_rate', 28),
              ('total_profit_growth_rate', 29)
              ,)


# 设置佣金和印花税率
class stampDutyCommissionScheme(bt.CommInfoBase):
    '''
    本佣金模式下，买入股票仅支付佣金，卖出股票支付佣金和印花税.    
    '''
    params = (
        ('stamp_duty', 0.001),  # 印花税率
        ('commission', 0.0005),  # 佣金率
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        '''
        If size is greater than 0, this indicates a long / buying of shares.
        If size is less than 0, it idicates a short / selling of shares.
        '''
        print('self.p.commission', self.p.commission)
        if size > 0:  # 买入，不考虑印花税
            return size * price * self.p.commission * 100
        elif size < 0:  # 卖出，考虑印花税
            return - size * price * (self.p.stamp_duty + self.p.commission * 100)
        else:
            return 0

        # 编写策略


class momentum_factor_strategy(bt.Strategy):
    # interval-换仓间隔，stocknum-持仓股票数
    params = (("interval", 1), ("stocknum", 10),)

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('{}, {}'.format(dt.isoformat(), txt))

    def __init__(self):
        # 已清洗过的每日可用股票列表
        self.df_byday = stocklist_allA
        # 交易天数，用于判断是否交易
        self.bar_num = 0
        # 上次交易股票的列表
        self.last = []

        # 记录以往订单，在调仓日要全部取消未成交的订单
        self.order_list = []

    def prenext(self):

        self.next()

    def next(self):
        self.bar_num += 1
        print("当天日期:{}".format(str(self.datas[0].datetime.date(0))))
        if self.bar_num % self.p.interval == 0 and self.bar_num > 3 * self.p.interval and self.datas[0].datetime.date(
                0) < datetime.date(2020, 6, 25):
            current_date = self.datas[0].datetime.date(0)
            print("交易日日期:{}".format(str(self.datas[0].datetime.date(0))))
            prev_date = self.datas[0].datetime.date(-self.p.interval)
            stocklist = self.get_valid_list_day(current_date)
            stocklist_p = self.get_valid_list_day(prev_date)

            df_fac = self.get_df_fac(stocklist=stocklist, prev=0)
            df_fac = df_fac.dropna(axis=0, how='any')
            df_fac_p = self.get_df_fac(stocklist=stocklist_p, prev=1)
            df_fac_p = df_fac_p.dropna(axis=0, how='any')

            # 本期因子排列命名
            factor_columns = ['code', 'natural_log_of_market_cap', 'earnings_to_price_ratio', 'book_to_price_ratio',
                              'cash_earnings_to_price_ratio', 'sales_to_price_ratio', 'admin_expense_rate',
                              'book_leverage', 'cash_to_current_liability', 'current_ratio', 'eps_ttm',
                              'equity_to_fixed_asset_ratio', 'fixed_asset_ratio', 'gross_income_ratio',
                              'intangible_asset_ratio', 'market_leverage', 'operating_cost_to_operating_revenue_ratio',
                              'quick_ratio', 'roa_ttm', 'roe_ttm', 'net_invest_cash_flow_ttm', 'net_asset_growth_rate',
                              'net_profit_growth_rate', 'operating_profit_growth_rate', 'operating_revenue_growth_rate',
                              'net_operate_cashflow_growth_rate', 'total_asset_growth_rate', 'total_profit_growth_rate']
            df_fac.columns = factor_columns
            df_fac.index = df_fac.code.values

            df_fac_p.columns = factor_columns
            df_fac_p.index = df_fac_p.code.values

            # 处理不一致的索引
            diffIndex = df_fac_p.index.difference(df_fac.index)
            df_fac_p = df_fac_p.drop(diffIndex, errors='ignore')
            df_fac = df_fac.drop(diffIndex, errors='ignore')
            diffIndex = df_fac.index.difference(df_fac_p.index)
            df_fac_p = df_fac_p.drop(diffIndex, errors='ignore')
            df_fac = df_fac.drop(diffIndex, errors='ignore')

            X_p = df_fac_p.drop(columns=['code'])
            X = df_fac.drop(columns=['code'])
            RealMV = df_fac[['natural_log_of_market_cap']]  # 假设这个为真实市场价值

            # 标准化因子数据
            rbX = RobustScaler()
            X_p = rbX.fit_transform(X_p)
            X = rbX.transform(X)

            # 使用上一期因子值来预测真实市场价值
            self.adaboost_model.fit(X_p, RealMV.values.ravel())
            PredMV = self.adaboost_model.predict(X)

            # 计算错误定价变量 M_i,t
            df_fac['M_i_t'] = PredMV / RealMV.values.ravel()

            # 按照错误定价变量排序
            df_fac.sort_values(by="M_i_t", inplace=True, ascending=False)

            # 选择错误定价最适合投资的前10只股票进行买入
            long_list = df_fac.head(self.p.stocknum)['code'].tolist()

            # 取消以往订单
            for o in self.order_list:
                self.cancel(o)
            self.order_list = []

            # 平仓不再持有的股票
            for i in self.last:
                if i not in long_list:
                    d = self.getdatabyname(i)
                    print('sell 平仓', d._name, self.getposition(d).size)
                    o = self.close(data=d)
                    self.order_list.append(o)

            self.log('当前总市值 %.2f' % (self.broker.getvalue()))
            total_value = self.broker.getvalue()

            # 买入选定的股票
            if len(long_list):
                buypercentage = (1 - 0.05) / len(long_list)
                targetvalue = buypercentage * total_value
                for d in long_list:
                    data = self.getdatabyname(d)
                    size = int(abs(targetvalue / data.open[1] // 100 * 100))
                    o = self.order_target_size(data=d, target=size)
                    self.order_list.append(o)

            self.last = long_list

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f, Size: %.2f, Stock: %s' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm,
                     order.executed.size,
                     order.data._name))
            else:
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f, Size: %.2f, Stock: %s' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm,
                          order.executed.size,
                          order.data._name))

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log('TRADE PROFIT, GROSS %.2f, NET %.2f' %
                     (trade.pnl, trade.pnlcomm))

    def get_df_fac(self, stocklist, prev=0):
        df_fac = pd.DataFrame(
            columns=['code', 'natural_log_of_market_cap', 'earnings_to_price_ratio', 'book_to_price_ratio',
                     'cash_earnings_to_price_ratio', 'sales_to_price_ratio', 'admin_expense_rate',
                     'book_leverage', 'cash_to_current_liability', 'current_ratio', 'eps_ttm',
                     'equity_to_fixed_asset_ratio', 'fixed_asset_ratio', 'gross_income_ratio',
                     'intangible_asset_ratio', 'market_leverage', 'operating_cost_to_operating_revenue_ratio',
                     'quick_ratio', 'roa_ttm', 'roe_ttm', 'net_invest_cash_flow_ttm', 'net_asset_growth_rate',
                     'net_profit_growth_rate', 'operating_profit_growth_rate', 'operating_revenue_growth_rate',
                     'net_operate_cashflow_growth_rate', 'total_asset_growth_rate', 'total_profit_growth_rate'])
        for stock in stocklist:
            data = self.getdatabyname(stock)
            N = []
            for i in range(self.p.interval):
                N.append([
                    data.natural_log_of_market_cap[(-i - prev * self.p.interval)],
                    data.earnings_to_price_ratio[(-i - prev * self.p.interval)],
                    data.book_to_price_ratio[(-i - prev * self.p.interval)],
                    data.cash_earnings_to_price_ratio[(-i - prev * self.p.interval)],
                    data.sales_to_price_ratio[(-i - prev * self.p.interval)],
                    data.admin_expense_rate[(-i - prev * self.p.interval)],
                    data.book_leverage[(-i - prev * self.p.interval)],
                    data.cash_to_current_liability[(-i - prev * self.p.interval)],
                    data.current_ratio[(-i - prev * self.p.interval)],
                    data.eps_ttm[(-i - prev * self.p.interval)],
                    data.equity_to_fixed_asset_ratio[(-i - prev * self.p.interval)],
                    data.fixed_asset_ratio[(-i - prev * self.p.interval)],
                    data.gross_income_ratio[(-i - prev * self.p.interval)],
                    data.intangible_asset_ratio[(-i - prev * self.p.interval)],
                    data.market_leverage[(-i - prev * self.p.interval)],
                    data.operating_cost_to_operating_revenue_ratio[(-i - prev * self.p.interval)],
                    data.quick_ratio[(-i - prev * self.p.interval)],
                    data.roa_ttm[(-i - prev * self.p.interval)],
                    data.roe_ttm[(-i - prev * self.p.interval)],
                    data.net_invest_cash_flow_ttm[(-i - prev * self.p.interval)],
                    data.net_asset_growth_rate[(-i - prev * self.p.interval)],
                    data.net_profit_growth_rate[(-i - prev * self.p.interval)],
                    data.operating_profit_growth_rate[(-i - prev * self.p.interval)],
                    data.operating_revenue_growth_rate[(-i - prev * self.p.interval)],
                    data.net_operate_cashflow_growth_rate[(-i - prev * self.p.interval)],
                    data.total_asset_growth_rate[(-i - prev * self.p.interval)],
                    data.total_profit_growth_rate[(-i - prev * self.p.interval)]
                ])
            N = np.mean(N, axis=0)
            new = pd.DataFrame({'code': stock,
                                'natural_log_of_market_cap': N[0],
                                'earnings_to_price_ratio': N[1],
                                'book_to_price_ratio': N[2],
                                'cash_earnings_to_price_ratio': N[3],
                                'sales_to_price_ratio': N[4],
                                'admin_expense_rate': N[5],
                                'book_leverage': N[6],
                                'cash_to_current_liability': N[7],
                                'current_ratio': N[8],
                                'eps_ttm': N[9],
                                'equity_to_fixed_asset_ratio': N[10],
                                'fixed_asset_ratio': N[11],
                                'gross_income_ratio': N[12],
                                'intangible_asset_ratio': N[13],
                                'market_leverage': N[14],
                                'operating_cost_to_operating_revenue_ratio': N[15],
                                'quick_ratio': N[16],
                                'roa_ttm': N[17],
                                'roe_ttm': N[18],
                                'net_invest_cash_flow_ttm': N[19],
                                'net_asset_growth_rate': N[20],
                                'net_profit_growth_rate': N[21],
                                'operating_profit_growth_rate': N[22],
                                'operating_revenue_growth_rate': N[23],
                                'net_operate_cashflow_growth_rate': N[24],
                                'total_asset_growth_rate': N[25],
                                'total_profit_growth_rate': N[26]}, index=[1])
            df_fac = df_fac.append(new, ignore_index=True)
        return df_fac

    def get_valid_list_day(self, current_date):
        self.df_byday['date'] = pd.to_datetime(self.df_byday['date'])
        current_date = datetime.datetime.strptime(str(current_date), '%Y-%m-%d')
        df_day = self.df_byday[self.df_byday['Date'] == current_date]
        stocklist = literal_eval(df_day['stocklist'].tolist()[0])
        return stocklist


##########################
# 主程序开始
##########################

import time
import datetime
import backtrader as bt

begin_time = time.time()
cerebro = bt.Cerebro(stdstats=False)

# 考虑印花税和佣金印花税为单边千分之一，佣金设为万五
comminfo = stampDutyCommissionScheme(stamp_duty=0.001, commission=0.0005)
cerebro.broker.addcommissioninfo(comminfo)

for s in stocklist_allA:
    df_stock = get_stock_data(s)
    #print(f"股票代码 {s} 的数据集大小: {len(df_stock)}")
    min_date = df_stock['date'].min()
    max_date = df_stock['date'].max()
    if len(df_stock) < 69:
        continue  # 跳过没有数据的股票

    feed = Addmoredata(dataname=df_stock, plot=False,
                       fromdate=min_date, todate=max_date)
    cerebro.adddata(feed, name=s)

cerebro.broker.setcash(1000000.0)
cerebro.broker.set_checksubmit(False)

print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

startcash = 1000000.0
cerebro.addstrategy(momentum_factor_strategy)
cerebro.addobserver(bt.observers.Value)

# 添加Analyzer
cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.01, annualize=True, _name='sharpe_ratio')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.Returns, tann=252, _name='returns')
cerebro.addanalyzer(bt.analyzers.VWR, _name='volatility')


thestrats = cerebro.run()
thestrat = thestrats[0]

# 获取分析器结果
sharpe_ratio = thestrat.analyzers.sharpe_ratio.get_analysis().get('sharperatio', None)
drawdown = thestrat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', None)
returns = thestrat.analyzers.returns.get_analysis().get('rnorm100', None)
volatility = thestrat.analyzers.volatility.get_analysis().get('stddev', None)

# 输出结果
print(f'Sharpe Ratio: {sharpe_ratio}')
print(f'Max DrawDown: {drawdown}%')
print(f'Annualized Return: {returns}%')
print(f'Annualized Volatility: {volatility}%')

# 获取回测结束后的总资金
portvalue = cerebro.broker.getvalue()
pnl = portvalue - startcash

# 打印结果
print(f'总资金: {round(portvalue, 2)}')
print(f'净收益: {round(pnl, 2)}')

end_time = time.time()
print(f"一共使用时间为: {end_time - begin_time} 秒")

# 打印各个分析器内容（可选）
for a in thestrat.analyzers:
    a.print()

# 绘制策略图表
cerebro.plot()
