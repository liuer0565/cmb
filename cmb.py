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
# 评估
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from scipy.stats import ks_2samp

# 训练样本
# 读取log数据
df_log=pd.read_csv('E:/cmb/train/train_log.csv',sep='\t')
df_log0=df_log[df_log['TCH_TYP']==0]
df_log0_gp=df_log0.groupby('USRID', as_index=False)[['OCC_TIM']].count()
df_log0_gp.rename(columns={'OCC_TIM': 'TCH_TYP0'}, inplace = True)
# df_log1=df_log[df_log['TCH_TYP']==1]
df_log2=df_log[df_log['TCH_TYP']==2]
df_log2_gp=df_log2.groupby('USRID', as_index=False)[['OCC_TIM']].count()
df_log2_gp.rename(columns={'OCC_TIM': 'TCH_TYP2'}, inplace = True)
df_log_merge=pd.merge(df_log0_gp, df_log2_gp, how='outer', on=['USRID'])
df_log=df_log_merge.fillna(0)
# pivot_table操作，暂时用不到
# cnt2=pd.pivot_table(cnt, values='OCC_TIM', index='USRID', columns='TCH_TYP', fill_value=0)
# 读取agg数据
df_agg=pd.read_csv('E:/cmb/train/train_agg.csv',sep='\t')
# 读取flg数据
df_flg=pd.read_csv('E:/cmb/train/train_flg.csv',sep='\t')
# merge
df_merge1=pd.merge(df_flg, df_agg, how='left', on=['USRID'])
df_merge2=pd.merge(df_merge1, df_log, how='left', on=['USRID'])
df=df_merge2.fillna(0)

# 描述性分析
# df.describe(include='all').to_excel('df_describe_cma.xls')

# 去掉id变量
df_noid = df.drop('USRID', 1)

# 变量衍生
df_derive=df_noid
# for i in df_derive.columns[1:]:
#     df_derive[str(i) + '_1'] = df_derive[i].apply(lambda x: 0 if isnan(x) else 1)
# # df_derive=df_derive.fillna(0)

# 缺失值补0
df_miss=df_derive.fillna(0)

# 正态标准化
x_std=StandardScaler().fit_transform(df_miss.iloc[:, 1:33])
y_std=df_miss.iloc[:, 0]

# 降维
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(x_std,y_std)
x_std_lda = lda.transform(x_std)

# 第二种调参处理方法：penalty='l1', C=1000.0, class_weight='balanced', random_state=0

# 训练样本和测试样本
train_data_para_2, test_data_para_2, train_target_para_2, test_target_para_2 = train_test_split(x_std_lda, y_std, test_size=0.30, random_state=2018, stratify=y_std)
get_ks = lambda y_pred,y_true: ks_2samp(y_pred[y_true==1], y_pred[y_true!=1]).statistic

# 训练
# 1、逻辑回归
clf_logistic_para_2 = linear_model.LogisticRegression(class_weight='balanced')
clf_logistic_para_2.fit(train_data_para_2, train_target_para_2)
logistic_train_target_para_2 = clf_logistic_para_2.predict(train_data_para_2)
print('调参之后-逻辑回归训练auc', metrics.roc_auc_score(train_target_para_2, logistic_train_target_para_2))
print('调参之后-逻辑回归训练ks', get_ks(logistic_train_target_para_2,train_target_para_2))

# 测试
# 1、逻辑回归
logistic_test_target_para_2 = clf_logistic_para_2.predict(test_data_para_2)
print('调参之后-逻辑回归测试auc', metrics.roc_auc_score(test_target_para_2, logistic_test_target_para_2))
print('调参之后-逻辑回归测试ks', get_ks(logistic_test_target_para_2,test_target_para_2))

# Kfold
print('调参之后-逻辑回归测试auc得分-5折交叉验证', cross_val_score(clf_logistic_para_2, x_std_lda, y_std, cv=5, scoring='roc_auc').mean())

# 验证样本
# 读取log数据
df1_log=pd.read_csv('E:/cmb/test/test_log.csv',sep='\t')
df1_log0=df1_log[df1_log['TCH_TYP']==0]
df1_log0_gp=df1_log0.groupby('USRID', as_index=False)[['OCC_TIM']].count()
df1_log0_gp.rename(columns={'OCC_TIM': 'TCH_TYP0'}, inplace = True)
# df1_log1=df1_log[df1_log['TCH_TYP']==1]
df1_log2=df1_log[df1_log['TCH_TYP']==2]
df1_log2_gp=df1_log2.groupby('USRID', as_index=False)[['OCC_TIM']].count()
df1_log2_gp.rename(columns={'OCC_TIM': 'TCH_TYP2'}, inplace = True)
df1_log_merge=pd.merge(df1_log0_gp, df1_log2_gp, how='outer', on=['USRID'])
df1_log=df1_log_merge.fillna(0)
# 读取agg数据
df1_agg=pd.read_csv('E:/cmb/test/test_agg.csv',sep='\t')
# merge
df1_merge=pd.merge(df1_log, df1_agg, how='right', on=['USRID'])
df1=df1_merge.fillna(0)

# 去掉id变量
# df1_noid = df1.drop('USRID', 1)
df1_noid = df1

# 变量衍生
df1_derive=df1_noid

# 缺失值补0
df1_miss=df1_derive.fillna(0)

# 正态标准化
x1_std=StandardScaler().fit_transform(df1_miss.iloc[:, 1:33])

# 降维（采用跟训练一样的降维模型）
x1_std_lda = lda.transform(x1_std)

# 预测（采用跟训练一样的分类模型）
logistic_val_target_para_2 = clf_logistic_para_2.predict(x1_std_lda)

# 导出csv
# df1_x_y=pd.DataFrame([df1['USRID'],pd.Series(logistic_val_target_para_2)])
a = df1['USRID']
b = pd.Series(logistic_val_target_para_2)
c = pd.DataFrame({'USRID':a, 'RST':b})

c_id = c.USRID
d = c.drop('USRID',axis=1)
d.insert(0,'USRID',c_id)

d.to_csv('C:/Users/xujianhua/PycharmProjects/cmb/submit_sample.csv', sep='\t', index=False)