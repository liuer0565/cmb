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

def kh_xgboost(df_train,df_test):
    # 训练样本
    # 去掉id变量
    df_noid = df_train.drop('USRID', 1)
    # 获取自变量和因变量
    x, y = df_noid.iloc[:, 1:], df_noid.iloc[:, 0]
    # 标准化
    x_std = StandardScaler().fit_transform(x)
    pca = PCA(n_components=330)
    x_std_pca = pca.fit_transform(x_std)
    x_train, x_test, y_train, y_test = train_test_split(x_std_pca, y, test_size=0.2, random_state=123, stratify=y)
    clf = XGBClassifier(booster='gbtree',
                        objective='binary:logistic',  # 指定损失函数
                        learning_rate=0.02,  # 学习率(即步长)
                        n_estimators=600,  # 树的个数(太多容易过拟合)
                        max_depth=5,  # 树的深度(太深容易过拟合)
                        min_child_weight=3,  # 叶子节点最小权重
                        gamma=0.3,  # 惩罚项中叶子结点个数前的参数
                        subsample=0.8,  #  随机选择80%样本建立决策树
                        colsample_btree=0.7,  # 随机选择70%特征建立决策树
                        reg_alpha=0.6,  # 正则化参数
                        reg_lambda=0.6,  # 正则化参数
                        scale_pos_weight=1,  # 解决样本个数不平衡的问题
                        random_state=27,  # 随机数种子
                        silent=0,  # 设置成1则没有运行信息输出，最好是设置为0
                        eta=0.02,  # 如同学习率
                        nthread=7,  # cpu 线程数
                        eval_metric='auc'  # 评价方式
                        )
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
    df = kh_xgboost(df_train, df_test)
    df.to_csv('../xjh_test_xgboost_result.csv', sep='\t', index=False)

if __name__ == '__main__':
    kh_main()