#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# FileName: parse_lude_backtest_to_db.py
# Time: 2025/5/29 18:07
import os
import pymysql

import pandas as pd

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'quant_voyager'
}

path = './2018.1.2-2025.5.26单因子回测'


def proces_single_file(file_path):
    """
    处理单个 Excel 文件，将不同 sheet 的数据保存到对应的数据库表中。
    """
    # 建立数据库连接
    try:
        db_connection = pymysql.connect(**DB_CONFIG)
        print("数据库连接成功")
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return

    # 获取数据库游标
    cursor = db_connection.cursor()

    try:
        xls = pd.ExcelFile(file_path)
    except Exception as e:
        print(f"无法读取 Excel 文件: {e}")
        db_connection.close()
        return

    filename = file_path.split('/')[-1]
    # 提取“正相关”或“负相关”
    correlation_type = filename.split('-')[0]
    # 提取因子
    factor = filename.split('-')[1].replace('回测报告.xlsx', '')
    print(correlation_type, factor)

    # 处理 "回测结果" sheet
    if "回测结果" in xls.sheet_names:
        # 读取回测结果 sheet
        df_backtest = pd.read_excel(xls, "回测结果")
        # 跳过第一行，然后循环遍历每一行
        for index, row in df_backtest.iloc[0:].iterrows():
            # print(row)
            # SQL 插入语句
            sql = """
            INSERT INTO lude_backtest_results (correlation_type, factor, strategy_combination, cumulative_assets, total_return, annualized_return, max_drawdown, sharpe_ratio,
            sortino_ratio, calmar_ratio, daily_turnover, trading_cycles, profitable_cycles, loss_cycles,win_rate, profit_loss_ratio, average_cycle_return,
            max_single_cycle_profit,max_single_cycle_loss)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)  # 修正占位符数量为19个
            """
            strategy = row['策略组合']
            try:
                result = cursor.execute(sql, (
                    correlation_type,
                    factor,
                    row['策略组合'],
                    row['累计资产'] if pd.notnull(row['累计资产']) else None,
                    row['总收益率'] if pd.notnull(row['总收益率']) else None,
                    row['年化收益率'] if pd.notnull(row['年化收益率']) else None,
                    float(row['最大回撤率'].rstrip('%')) / 100 if pd.notnull(row['最大回撤率']) and '%' in str(
                        row['最大回撤率']) else None,
                    row['夏普比率'] if pd.notnull(row['夏普比率']) else None,
                    row['索提诺比率'] if pd.notnull(row['索提诺比率']) else None,
                    row['卡玛比率'] if pd.notnull(row['卡玛比率']) else None,
                    row['日均换手率'] if pd.notnull(row['日均换手率']) else None,
                    row['交易周期数'] if pd.notnull(row['交易周期数']) else None,
                    row['盈利周期数'] if pd.notnull(row['盈利周期数']) else None,
                    row['亏损周期数'] if pd.notnull(row['亏损周期数']) else None,
                    row['胜率'] if pd.notnull(row['胜率']) else None,
                    row['盈亏比'] if pd.notnull(row['盈亏比']) else None,
                    row['平均每周期收益'] if pd.notnull(row['平均每周期收益']) else None,
                    row['最大单周期盈利'] if pd.notnull(row['最大单周期盈利']) else None,
                    row['最大单周期亏损'] if pd.notnull(row['最大单周期亏损']) else None
                ))
                # print("回测结果数据插入成功")
                # print(result)
                db_connection.commit()
                print(f'{correlation_type} - {factor}， 保存 {strategy} 回测结果成功，result={result}')
            except Exception as e:
                print(f"插入回测结果数据失败: {e}")

    # 处理回报相关的 sheet（年度回报、月度回报、周度回报等）
    return_sheets = [name for name in xls.sheet_names if "回报" in name]
    for sheet_name in return_sheets:
        # 读取回报 sheet
        df_returns = pd.read_excel(xls, sheet_name)
        return_type = sheet_name.replace("回报", "")  # 提取回报类型，如“年度”
        # 遍历每一行数据并插入
        for index, row in df_returns.iterrows():
            sql = """
            INSERT INTO lude_returns (correlation_type, factor, return_type, date, strategy_return, benchmark_return, excess_return)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            try:
                result = cursor.execute(sql, (
                    correlation_type,
                    factor,
                    return_type,
                    row[return_type],
                    row['策略收益'],
                    row['基准收益'],
                    row['超额收益']
                ))
                db_connection.commit()
                print(
                    f'{correlation_type} - {factor}， 保存 {return_type} {row[return_type]} 回报结果成功，result={result}')
            except Exception as e:
                print(f"插入 {sheet_name} 数据失败: {e}")
        print(f"{sheet_name} 数据插入成功")

    # 处理 "持仓详情" sheet
    if "持仓详情" in xls.sheet_names:
        # 读取持仓明细 sheet
        df_holdings = pd.read_excel(xls, "持仓详情")
        # 遍历每一行数据并插入
        for index, row in df_holdings.iterrows():
            sql = """
            INSERT INTO lude_holdings (correlation_type, factor, date, holding, holding_quantity, buy_quantity, sell_quantity, turnover, next_cycle_pct)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            try:
                result = cursor.execute(sql, (
                    correlation_type,
                    factor,
                    row['选债日(收盘后)'],
                    row['持仓标的'],
                    row['持有数量'],
                    row['买入数量'],
                    row['卖出数量'],
                    row['换手率'],
                    row['下周期持仓涨跌幅']
                ))
                db_connection.commit()
                print(f'{correlation_type} - {factor}， 保存持仓明细成功，result={result}')
            except Exception as e:
                print(f"插入持仓明细数据失败: {e}")
        print("持仓明细数据插入成功")

    # 关闭数据库连接
    db_connection.close()
    print("数据库连接已关闭")


if __name__ == '__main__':
    # 读取 path 下的 xlsx 文件
    for f in os.listdir(path):
        if f.endswith('.xlsx'):
            # 获取文件的全路径
            file_path = os.path.join(path, f)
            # print(file_path)
            proces_single_file(file_path)
            # df = pd.read_excel(os.path.join(path, f))
            # print(df)
