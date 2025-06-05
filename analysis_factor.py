#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# FileName: analysis_factor.py
# Time: 2025/6/4 18:01
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.fft import fft
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine
import seaborn as sns  # 用于可视化相关性矩阵

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统字体
plt.rcParams['axes.unicode_minus'] = False

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'quant_voyager'
}

engine = None

effective_factors = ['转股溢价率_负相关', '修正溢价率_负相关', '5日均价_负相关', '理论偏离度_负相关', '最高价_负相关',
                     '最低价_负相关', '正股市销率_正相关', '剩余市值_负相关', '纯债溢价率_负相关', '到期收益率_正相关',
                     '双低_负相关', '开盘价_负相关', '正股市盈率_正相关', '收盘价_负相关', '正股总市值_负相关',
                     '5日超额涨跌幅_负相关', '正股5日涨跌幅_正相关', '正股流通市值_负相关', '上市天数_负相关',
                     '前收盘价_负相关']


def process_data(df):
    # 如果有 date 字段，则转换日期格式
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    # 设置索引为 id
    df.set_index('id', inplace=True)
    df['factor'] = df['factor'] + '_' + df['correlation_type']
    return df


def analyze_factor(df):
    df = df[['factor', 'date', 'next_cycle_pct']]
    # print(df)
    # print(df['factor'].unique())

    # analyze_time_series(df)
    # rolling_window_statistics(df)
    # analyze_season(df)
    # analyze_effectiveness(df)
    analyze_correlation(df)


def analyze_time_series(df):
    # for factor_name in df['factor'].unique():
    for factor_name in effective_factors:
        # 筛选当前因子的数据
        df_factor = df[df['factor'] == factor_name].copy()

        # 绘制回报时间序列图
        plt.figure(figsize=(12, 6))
        plt.plot(df_factor['date'], df_factor['next_cycle_pct'])
        plt.title(f'{factor_name} 对应的下一周期回报时间序列')
        plt.xlabel('时间')
        plt.ylabel('回报率')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def rolling_window_statistics(df, window_size=12):
    for factor_name in effective_factors:
        df_factor = df[df['factor'] == factor_name].copy()

        # 计算滚动均值和滚动标准差
        df_factor['rolling_mean'] = df_factor['next_cycle_pct'].rolling(window=window_size).mean()
        df_factor['rolling_std'] = df_factor['next_cycle_pct'].rolling(window=window_size).std()

        # 绘制滚动均值和标准差图
        plt.figure(figsize=(12, 6))
        plt.plot(df_factor['date'], df_factor['rolling_mean'], label='滚动均值')
        plt.plot(df_factor['date'], df_factor['rolling_std'], label='滚动标准差')
        plt.title(f'{factor_name} 的滚动均值和标准差')
        plt.xlabel('时间')
        plt.ylabel('值')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def analyze_ranking(df):
    df['rank'] = df.groupby('date')['strategy_return'].rank(ascending=False)

    plt.figure(figsize=(16, 8))

    # 绘制每个因子的排名随时间变化曲线
    for factor_name in effective_factors:
        df_factor = df[df['factor'] == factor_name].copy()
        # print(f'{factor_name}在{df_factor['date']}的排名为{df_factor['rank']}')
        plt.plot(df_factor['date'], df_factor['rank'], label=factor_name, alpha=0.7,
                 linewidth=1)
        # plt.figure(figsize=(12, 6))
        # plt.plot(df_factor['date'], df_factor['rank'])
        # plt.title(f'{factor_name} 的回报排名变化')
        # plt.xlabel('时间')
        # plt.ylabel('排名')
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.show()

    plt.title('所有因子的回报排名变化')
    plt.xlabel('时间')
    plt.ylabel('排名')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # 添加图例（自动根据label生成）
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), ncol=2)
    plt.show()


# 季节性分析
def analyze_season(df):
    for factor_name in effective_factors:
        df_factor = df[df['factor'] == factor_name].copy()

        # 提取月份和年份
        df_factor['month'] = df_factor['date'].dt.month
        df_factor['year'] = df_factor['date'].dt.year

        # 按月份分组，计算每月的平均回报
        monthly_returns = df_factor.groupby('month')['next_cycle_pct'].mean()

        # 绘制季节性图
        plt.figure(figsize=(12, 6))
        monthly_returns.plot(kind='bar')
        plt.title(f'{factor_name} 的月度平均回报')
        plt.xlabel('月份')
        plt.ylabel('平均回报率')
        plt.tight_layout()
        plt.show()


def analyze_effectiveness(df):
    # 按因子分组 -> 创建不同因子的分析数据集
    grouped = df.groupby('factor')

    # 定义存储结果的列表
    results = []

    # 核心分析逻辑包含四个维度：
    for factor, group in grouped:
        if factor not in effective_factors:
            continue
        group = group.sort_values(by='date')
        # 1. 趋势分析（线性回归斜率）
        # 将时间序列转换为数值特征
        X = np.arange(len(group)).reshape(-1, 1)
        y = group['next_cycle_pct'].values
        # 用线性回归拟合时间趋势
        model = LinearRegression().fit(X, y)

        # 提取斜率作为趋势强度
        trend = model.coef_[0]

        # 2. 周期性分析（傅里叶变换）
        fft_values = fft(y)
        dominant_frequency = np.argmax(np.abs(fft_values[1:])) + 1

        # 3. 滚动窗口统计（12周）
        rolling_mean = group['next_cycle_pct'].rolling(window=12).mean().dropna().mean()

        # 滚动窗口统计 - 滚动标准差
        rolling_std = group['next_cycle_pct'].rolling(window=12).std().dropna().mean()

        # 将结果添加到列表
        results.append({
            'factor': factor,
            'trend': trend,
            'dominant_frequency': dominant_frequency,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std
        })

    # 将结果列表转换为DataFrame
    results_df = pd.DataFrame(results)

    # 4. 因子有效性评估 |趋势| / 滚动标准差 （简单示例：趋势绝对值大、滚动标准差小认为有效）
    results_df['effectiveness_score'] = np.abs(results_df['trend']) / results_df['rolling_std']

    # 按有效性得分降序排序
    results_df = results_df.sort_values(by='effectiveness_score', ascending=False)

    print(results_df.head(10))


def analyze_returns(df):
    # analyze_ranking(df[df['return_type'] == '月度'].copy())

    df = df[df['return_type'] == '周度'].copy()
    # print(df)

    long_term_factors = []
    short_term_factors = []

    analyze_week_returns(df, '2021-01-01', short_term_factors)
    analyze_week_returns(df, '2018-01-01', long_term_factors)

    print(f'短期因子: {short_term_factors}')
    print(f'长期因子: {long_term_factors}')

    # 分别取交集和差集
    intersection = list(set(short_term_factors) & set(long_term_factors))
    short_difference = list(set(short_term_factors) - set(long_term_factors))
    long_difference = list(set(long_term_factors) - set(short_term_factors))
    print(f'交集: {intersection}')
    print(f'short 差集: {short_difference}')
    print(f'long 差集: {long_difference}')

    effective_factors = intersection
    print(f'有效因子: {effective_factors}')


def analyze_correlation(df):
    # 筛选出 effective_factors 对应的因子数据
    df = df[df['factor'].isin(effective_factors)]

    # 将数据按日期和因子进行透视
    pivot_df = df.pivot(index='date', columns='factor', values='next_cycle_pct')

    # 计算相关性矩阵
    correlation_matrix = pivot_df.corr()

    # 打印相关性矩阵
    print(correlation_matrix)

    # 找出相关性 >0.7 的因子对
    strong_correlations = []

    # 遍历相关性矩阵的上三角部分（不包括对角线）
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            factor1 = correlation_matrix.columns[i]
            factor2 = correlation_matrix.columns[j]
            correlation = correlation_matrix.iloc[i, j]

            # 筛选出绝对值 >0.7 的强相关因子对
            if abs(correlation) > 0.7:
                strong_correlations.append({
                    'factor1': factor1,
                    'factor2': factor2,
                    'correlation': correlation
                })

    # 将结果转换为DataFrame并排序
    if strong_correlations:
        strong_correlations_df = pd.DataFrame(strong_correlations)
        strong_correlations_df = strong_correlations_df.sort_values(
            by='correlation',
            key=abs,
            ascending=False
        )
        print("\n强相关因子对 (|相关性| > 0.7):")
        print(strong_correlations_df)
    else:
        print("\n没有发现相关性 >0.7 的因子对")

    # 可视化相关性矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Effective Factors Correlation Matrix')
    plt.show()


def analyze_week_returns(df, date, factors):
    df = df[df['date'] >= date]

    # 按因子分组 -> 创建不同因子的分析数据集
    grouped = df.groupby('factor')
    for factor, group in grouped:
        group = group.sort_values(by='date')
        # 计算累计回报率
        group['strategy_cum_return'] = (1 + group['strategy_return']).cumprod()
        group['benchmark_cum_return'] = (1 + group['benchmark_return']).cumprod()

        if group.iloc[-1]['strategy_cum_return'] < group.iloc[-1]['benchmark_cum_return']:
            # print(
            # f'{factor} 的周度累计收益率 {group.iloc[-1]['strategy_cum_return']}小于基准累计收益率 {group.iloc[-1]['benchmark_cum_return']}')
            continue

        print(
            f'{factor} 的周度累计收益率 {group.iloc[-1]['strategy_cum_return']}大于基准累计收益率 {group.iloc[-1]['benchmark_cum_return']}')
        factors.append(factor)
        # 绘制累计收益率曲线
        # plt.figure(figsize=(12, 6))
        # plt.plot(group['date'], group['strategy_cum_return'])
        # plt.title(f'{factor} 的周度累计收益率')
        # plt.xlabel('时间')
        # plt.ylabel('累计收益率')
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.show()


if __name__ == '__main__':

    # 建立数据库连接
    try:
        engine = create_engine(
            f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")
        # backtest_sql = "SELECT * FROM lude_backtest_results"
        # df_backtest = pd.read_sql(backtest_sql, engine)
        # df_backtest = process_data(df_backtest)
        # print(df_backtest)

        returns_sql = "SELECT * FROM lude_returns"
        df_returns = pd.read_sql(returns_sql, engine)
        df_returns = process_data(df_returns)
        # print(df_returns)
        analyze_returns(df_returns)

        holdings_sql = "SELECT * FROM lude_holdings"
        df_holdings = pd.read_sql(holdings_sql, engine)
        df_holdings = process_data(df_holdings)
        # print(df_holdings)
        analyze_factor(df_holdings)
    except Exception as e:
        print(f"数据库连接失败: {e}")
    finally:
        # 关闭数据库连接
        if engine:
            engine.dispose()
            print("数据库连接已关闭")
