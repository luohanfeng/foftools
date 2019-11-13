"""
工具函数
"""
import tushare as ts
import numpy as np
import pandas as pd
import xlrd
from pyfolio.timeseries import gen_drawdown_table, perf_stats


def read_sheet(file_path):
    """
    读入产品数据
    excel格式，一个sheet对应一个产品
    sheet格式，第一列为日期，第二列为累计净值
    :param file_path:
    :return:
    """
    excel = xlrd.open_workbook(file_path)
    sheet_list = excel.sheet_names()
    df_list = []
    for each in sheet_list:
        df = pd.read_excel(file_path, each, header=0)
        if df.empty:
            continue
        print(f'读取{df.columns}')
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.iloc[:, 0].name, drop=True, inplace=True)
        df.drop_duplicates(inplace=True)
        df = df.resample('D').asfreq()
        df_list.append(df)
    total = pd.concat(df_list, axis=1)
    trade_list = ts.pro_api().trade_cal(exchange='SSE', start_date=total.index.min().strftime('%Y%m%d'),
                                        end_date=total.index.max().strftime('%Y%m%d'))
    trade_list = trade_list[trade_list['is_open'] == 1]  # 交易日
    total = total[total.index.isin(pd.to_datetime(trade_list['cal_date']))]
    total = total.apply(pd.Series.interpolate)
    return total


def value_to_return(se):
    """
    根据净值，计算日收益率
    :param se:
    :return:
    """
    # 空值不计算，总表中仍保留空值
    use_se = se.copy()
    use_se = use_se / use_se.shift(1) - 1
    use_se[se.isnull()] = np.nan
    return use_se


def weight_by_params(se, weight):
    """
    设置权重，有缺失值的产品不参与加权，并重新将参与加权的产品权重归一化，保持产品间比例不变。
    :param se:
    :param weight:
    :return:
    """
    if se.dropna().empty:
        return np.nan
    # na位置
    use_se = se[~se.isnull()]
    use_temp = weight[~se.isnull()]
    use_temp = use_temp/use_temp.sum()
    temp = np.dot(use_se, use_temp)  # -0.000152*0.384615+0.001594*0.307692+0.000567*0.307692=0.000606460932
    return temp


def stats_se(se):
    stats = perf_stats(se.dropna())
    stats['VaR'] = np.percentile(se.dropna(), 100 * 0.05)
    stats = stats.append(gen_drawdown_table(se, top=1).iloc[0, :])
    return stats


def stats_df(df):
    result = df.apply(stats_se)
    result['mean'] = result.T.mean()
    result['max_min'] = result.apply(lambda x: x.max() - x.min(), axis=1)
    return result