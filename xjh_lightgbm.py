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
import lightgbm as lgb
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

def kh_read():
    # 训练样本
    df_train = pd.read_csv('D:/0.python/Cmb/xjh_train.csv', sep='\t')
    # 验证样本
    # 读取log数据
    df_test = pd.read_csv('D:/0.python/Cmb/xjh_test.csv', sep='\t')

    return df_train,df_test

def kh_lightgbm(df_train,df_test):
    # 训练样本
    # 去掉id变量
    df_noid = df_train.drop('USRID', 1)
    # 获取自变量和因变量
    x, y = df_noid.iloc[:, 1:], df_noid.iloc[:, 0]
    # 标准化
    x_std = StandardScaler().fit_transform(x)
    pca = PCA(n_components=330)
    x_std_pca = pca.fit_transform(x_std)
    x_train, x_test, y_train, y_test = train_test_split(x_std_pca, y, test_size=0.1, random_state=123, stratify=y)
    # sklearn接口形式
    clf1 = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)
    clf1.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='l1', early_stopping_rounds=5)
    # 训练评估
    y_train_pred = clf1.predict(x_train)
    print('clf1网格搜索最佳训练auc(整数):', metrics.roc_auc_score(y_train, y_train_pred))
    # 测试评估
    y_test_pred = clf1.predict(x_test)
    print('clf1网格搜索最佳测试auc(整数):', metrics.roc_auc_score(y_test, y_test_pred))
    # 原生形式
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    params = {'task': 'train',
              'boosting_type': 'gbdt',  # 设置提升类型
              'objective': 'binary',  # 目标函数
              'metric': {'auc'},  # 评估函数
              'num_leaves': 10,  # 叶子节点数
              'learning_rate': 0.01,  # 学习速率
              'feature_fraction': 0.9,  # 建树的特征选择比例
              'bagging_fraction': 0.9,  # 建树的样本采样比例
              'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
              'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
              }
    clf2 = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=lgb_eval, early_stopping_rounds=50)
    # 训练评估
    y_train_pred = clf2.predict(x_train)
    print('clf2网格搜索最佳训练auc(整数):', metrics.roc_auc_score(y_train, y_train_pred))
    # 测试评估
    y_test_pred = clf2.predict(x_test)
    print('clf2网格搜索最佳测试auc(整数):', metrics.roc_auc_score(y_test, y_test_pred))

    # 验证样本
    x1_std = StandardScaler().fit_transform(df_test.iloc[:,1:])  # 标准化
    pca = PCA(n_components=330)
    x1_std_pca = pca.fit_transform(x1_std)  # 降维
    y1_pred = clf2.predict(x1_std_pca)  # 预测整数
    # y1_pro = clf.predict_proba(x1_std_pca)[:, 1]  # 预测小数
    # 创建dataframe
    a = df_test['USRID']
    b = pd.Series(y1_pred)
    c = pd.DataFrame({'USRID': a, 'RST': b})
    c_id = c.USRID
    d = c.drop('USRID', axis=1)
    d.insert(0, 'USRID', c_id)

    return d

def kh_main():
    df_train, df_test = kh_read()
    df = kh_lightgbm(df_train, df_test)
    df.to_csv('../xjh_test_lightgbm_result.csv', sep='\t', index=False)

if __name__ == '__main__':
    kh_main()