# 加载包
# 基础
from math import isnan
# 抽样
from sklearn.model_selection import train_test_split
# 特征选择
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# 降维
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 数据预处理
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# 建模
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from sklearn import naive_bayes
from sklearn import neural_network
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
# 评估
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from scipy.stats import ks_2samp
# 管道机制
from sklearn.pipeline import Pipeline
# 网格搜索
from sklearn.model_selection import GridSearchCV
# 计时
import time

def xjh_read():
    # 训练样本
    # 读取log数据
    df_log = pd.read_csv('D:/0.python/Cmb/Cmb_own/input/train/train_log.csv', sep='\t')
    # 读取agg数据
    df_agg = pd.read_csv('D:/0.python/Cmb/Cmb_own/input/train/train_agg.csv', sep='\t')
    # 读取flg数据
    df_flg = pd.read_csv('D:/0.python/Cmb/Cmb_own/input/train/train_flg.csv', sep='\t')
    # 验证样本
    # 读取log数据
    df1_log = pd.read_csv('D:/0.python/Cmb/Cmb_own/input/test/test_log.csv', sep='\t')
    # 读取agg数据
    df1_agg = pd.read_csv('D:/0.python/Cmb/Cmb_own/input/test/test_agg.csv', sep='\t')

    return df_log,df_agg,df_flg,df1_log,df1_agg

def xjh_feature(df_log):
    df_log['EVT_LBL_f'] = df_log['EVT_LBL'].str.split('-', expand=True)[0]  # 添加EVT_LBL前3位的分类
    df_log['EVT_LBL_ff'] = df_log['EVT_LBL'].str.split('-', expand=True)[1]  # 添加EVT_LBL中间3位的分类
    df_log['EVT_LBL_fff'] = df_log['EVT_LBL'].str.split('-', expand=True)[2]  # 添加EVT_LBL后3位的分类
    df_log['OCC_TIM_f'] = df_log['OCC_TIM'].str[11:13]  # 添加OCC_TIM的小时
    df_log['OCC_TIM_ff'] = df_log['OCC_TIM'].str[8:10]  # 添加OCC_TIM的天
    df_log['TCH_TYP_OCC_TIM'] = df_log['TCH_TYP'].map(str) + df_log['OCC_TIM'].str[11:13]  # 添加TCH_TYP拼接OCC_TIM
    df_log['TCH_TYP_EVT_LBL_f'] = df_log['TCH_TYP'].map(str) + df_log['EVT_LBL_f'].map(str)  # 添加TCH_TYP拼接EVT_LBL_f
    # OCC_TIM_ff
    df_log_gp8 = df_log.groupby(['USRID', 'OCC_TIM_ff'], as_index=False)[['OCC_TIM']].count()
    df_log_gp8_pivot = pd.pivot_table(df_log_gp8, values='OCC_TIM', index='USRID', columns='OCC_TIM_ff', fill_value=0)
    a = df_log_gp8_pivot.index.values
    b = df_log_gp8_pivot.values.T
    c = np.vstack((a, b))
    df_log_merge8 = pd.DataFrame(c.T)
    df_log_merge8.columns = df_log_merge8.columns.astype(str)
    df_log_merge8.columns = ['OCC_TIM_ff_' + i for ix, i in enumerate(df_log_merge8.columns)]
    df_log_merge8.rename(columns={'OCC_TIM_ff_0': 'USRID'}, inplace=True)
    # TCH_TYP拼接EVT_LBL_f
    df_log_gp7 = df_log.groupby(['USRID', 'TCH_TYP_EVT_LBL_f'], as_index=False)[['OCC_TIM']].count()
    df_log_gp7_pivot = pd.pivot_table(df_log_gp7, values='OCC_TIM', index='USRID', columns='TCH_TYP_EVT_LBL_f',
                                      fill_value=0)
    a = df_log_gp7_pivot.index.values
    b = df_log_gp7_pivot.values.T
    c = np.vstack((a, b))
    df_log_merge7 = pd.DataFrame(c.T)
    df_log_merge7.columns = df_log_merge7.columns.astype(str)
    df_log_merge7.columns = ['TCH_TYP_EVT_LBL_f_' + i for ix, i in enumerate(df_log_merge7.columns)]
    df_log_merge7.rename(columns={'TCH_TYP_EVT_LBL_f_0': 'USRID'}, inplace=True)
    # TCH_TYP拼接OCC_TIM
    df_log_gp6 = df_log.groupby(['USRID', 'TCH_TYP_OCC_TIM'], as_index=False)[['OCC_TIM']].count()
    df_log_gp6_pivot = pd.pivot_table(df_log_gp6, values='OCC_TIM', index='USRID', columns='TCH_TYP_OCC_TIM',
                                      fill_value=0)
    a = df_log_gp6_pivot.index.values
    b = df_log_gp6_pivot.values.T
    c = np.vstack((a, b))
    df_log_merge6 = pd.DataFrame(c.T)
    df_log_merge6.columns = df_log_merge6.columns.astype(str)
    df_log_merge6.columns = ['TCH_TYP_OCC_TIM_' + i for ix, i in enumerate(df_log_merge6.columns)]
    df_log_merge6.rename(columns={'TCH_TYP_OCC_TIM_0': 'USRID'}, inplace=True)
    # TCH_TYP分组
    df_log_gp = df_log.groupby(['USRID', 'TCH_TYP'], as_index=False)[['OCC_TIM']].count()
    df_log_gp_pivot = pd.pivot_table(df_log_gp, values='OCC_TIM', index='USRID', columns='TCH_TYP',
                                     fill_value=0)  # pivot_table操作
    a = df_log_gp_pivot.index.values
    b = df_log_gp_pivot.values.T
    c = np.vstack((a, b))
    df_log_merge1 = pd.DataFrame(c.T)
    df_log_merge1.columns = df_log_merge1.columns.astype(str)
    df_log_merge1.columns = ['TCH_TYP_' + i for ix, i in enumerate(df_log_merge1.columns)]
    df_log_merge1.rename(columns={'TCH_TYP_0': 'USRID'}, inplace=True)
    # EVT_LBL_f分组
    df_log_gp2 = df_log.groupby(['USRID', 'EVT_LBL_f'], as_index=False)[['OCC_TIM']].count()
    df_log_gp2_pivot = pd.pivot_table(df_log_gp2, values='OCC_TIM', index='USRID', columns='EVT_LBL_f', fill_value=0)
    a = df_log_gp2_pivot.index.values
    b = df_log_gp2_pivot.values.T
    c = np.vstack((a, b))
    df_log_merge2 = pd.DataFrame(c.T)
    df_log_merge2.columns = df_log_merge2.columns.astype(str)
    df_log_merge2.columns = ['EVT_LBL_f_' + i for ix, i in enumerate(df_log_merge2.columns)]
    df_log_merge2.rename(columns={'EVT_LBL_f_0': 'USRID'}, inplace=True)
    # EVT_LBL_ff分组
    df_log_gp3 = df_log.groupby(['USRID', 'EVT_LBL_ff'], as_index=False)[['OCC_TIM']].count()
    df_log_gp3_pivot = pd.pivot_table(df_log_gp3, values='OCC_TIM', index='USRID', columns='EVT_LBL_ff', fill_value=0)
    a = df_log_gp3_pivot.index.values
    b = df_log_gp3_pivot.values.T
    c = np.vstack((a, b))
    df_log_merge3 = pd.DataFrame(c.T)
    df_log_merge3.columns = df_log_merge3.columns.astype(str)
    df_log_merge3.columns = ['EVT_LBL_ff_' + i for ix, i in enumerate(df_log_merge3.columns)]
    df_log_merge3.rename(columns={'EVT_LBL_ff_0': 'USRID'}, inplace=True)
    # EVT_LBL_fff分组 未生成变量
    df_log_gp4 = df_log.groupby(['USRID', 'EVT_LBL_fff'], as_index=False)[['OCC_TIM']].count()
    df_log_gp4_pivot = pd.pivot_table(df_log_gp4, values='OCC_TIM', index='USRID', columns='EVT_LBL_fff', fill_value=0)
    # OCC_TIM_f分组
    df_log_gp5 = df_log.groupby(['USRID', 'OCC_TIM_f'], as_index=False)[['OCC_TIM']].count()
    df_log_gp5_pivot = pd.pivot_table(df_log_gp5, values='OCC_TIM', index='USRID', columns='OCC_TIM_f', fill_value=0)
    a = df_log_gp5_pivot.index.values
    b = df_log_gp5_pivot.values.T
    c = np.vstack((a, b))
    df_log_merge5 = pd.DataFrame(c.T)
    df_log_merge5.columns = df_log_merge5.columns.astype(str)
    df_log_merge5.columns = ['OCC_TIM_f_' + i for ix, i in enumerate(df_log_merge5.columns)]
    df_log_merge5.rename(columns={'OCC_TIM_f_0': 'USRID'}, inplace=True)
    # merge
    df_log_merge = pd.merge(df_log_merge1, df_log_merge2, how='inner', on=['USRID'])
    df_log_merge_merge = pd.merge(df_log_merge, df_log_merge3, how='inner', on=['USRID'])
    df_log_merge_merge_merge = pd.merge(df_log_merge_merge, df_log_merge5, how='inner', on=['USRID'])
    df_log_merge_merge_merge_merge = pd.merge(df_log_merge_merge_merge, df_log_merge6, how='inner', on=['USRID'])
    df_log_merge_merge_merge_merge_merge = pd.merge(df_log_merge_merge_merge_merge, df_log_merge7, how='inner',
                                                    on=['USRID'])
    df_log_merge_merge_merge_merge_merge_merge = pd.merge(df_log_merge_merge_merge_merge_merge, df_log_merge8,
                                                          how='inner', on=['USRID'])
    # 缺失值处理
    df_log = df_log_merge_merge_merge_merge_merge_merge.fillna(0)

    return df_log

def xjh_main():
    df_log, df_agg, df_flg, df1_log, df1_agg = xjh_read()
    # 训练样本
    df_log = xjh_feature(df_log)
    # merge
    df_merge1 = pd.merge(df_flg, df_agg, how='left', on=['USRID'])
    df_merge2 = pd.merge(df_merge1, df_log, how='left', on=['USRID'])
    # 缺失值处理
    df = df_merge2.fillna(0)
    df.to_csv('../xjh_train.csv', sep='\t', index=None)

    # 验证样本
    df1_log = xjh_feature(df1_log)
    # merge1
    df1_merge = pd.merge(df1_agg, df1_log, how='left', on=['USRID'])
    df1 = df1_merge.fillna(0)
    df1.to_csv('../xjh_test.csv', sep='\t', index=None)

if __name__ == '__main__':
    xjh_main()