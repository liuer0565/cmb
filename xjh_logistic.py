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

def kh_read():
    # 训练样本
    df_train = pd.read_csv('D:/0.python/Cmb/xjh_train.csv', sep='\t')
    # 验证样本
    # 读取log数据
    df_test = pd.read_csv('D:/0.python/Cmb/xjh_test.csv', sep='\t')

    return df_train,df_test

def kh_logistic(df_train,df_test):
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
    clf = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                                          intercept_scaling=1, class_weight='balanced', random_state=None,
                                          solver='liblinear', max_iter=100, multi_class='ovr', verbose=0,
                                          warm_start=False, n_jobs=1)
    # 训练
    clf.fit(x_train, y_train)
    # 训练评估
    y_train_pred = clf.predict(x_train)
    y_train_pro = clf.predict_proba(x_train)[:, 1]
    print('网格搜索最佳训练auc(整数):', metrics.roc_auc_score(y_train, y_train_pred))
    print('网格搜索最佳训练auc(小数):', metrics.roc_auc_score(y_train, y_train_pro))
    # 测试评估
    y_test_pred = clf.predict(x_test)
    y_test_pro = clf.predict_proba(x_test)[:, 1]
    print('网格搜索最佳测试auc(整数):', metrics.roc_auc_score(y_test, y_test_pred))
    print('网格搜索最佳测试auc(小数):', metrics.roc_auc_score(y_test, y_test_pro))

    # 验证样本
    x1_std = StandardScaler().fit_transform(df_test.iloc[:,1:])  # 标准化
    pca = PCA(n_components=330)
    x1_std_pca = pca.fit_transform(x1_std)  # 降维
    y1_pred = clf.predict(x1_std_pca)  # 预测整数
    y1_pro = clf.predict_proba(x1_std_pca)[:, 1]  # 预测小数
    # 创建dataframe
    a = df_test['USRID']
    b = pd.Series(y1_pro)
    c = pd.DataFrame({'USRID': a, 'RST': b})
    c_id = c.USRID
    d = c.drop('USRID', axis=1)
    d.insert(0, 'USRID', c_id)

    return d

def kh_main():
    df_train, df_test = kh_read()
    df = kh_logistic(df_train, df_test)
    df.to_csv('../xjh_test_logistic_result.csv', sep='\t', index=False)

if __name__ == '__main__':
    kh_main()