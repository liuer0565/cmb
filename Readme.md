# 第一部分：背景<br>
### 1.网址：https://www.datafountain.cn/competitions/287<br>
### 2.主题：消费金融场景下的用户购买预测<br>
### 3.主办方： 招商银行股份有限公司信用卡中心 & 招商银行信用卡中心<br>
### 4.数据下载：https://www.datafountain.cn/competitions/287/datasets<br>
### 5.数据背景：本次比赛提供的数据集包括训练数据与测试数据，训练数据共分为三部分<br>
#### (1)个人属性与信用卡消费数据：包含80000名信用卡客户的个人属性与信用卡消费数据，其中包含枚举型特征和数值型特征，均已转为数值并进行脱敏和标准。数据样例如下：<br>
USRID	V1	V2	v3	…	V30<br>
000001	-1.2212	0.4523	1.3251	…	-1.2212<br>
000002	-1.0987	0.0165	-1.0684	…	0.0925<br>
#### (2)APP操作行为日志：上述信用卡客户中，部分已绑定掌上生活APP的客户，在近一个月时间窗口内的所有点击行为日志。日志记录包含如下字段<br>
字段名	字段含义	说明<br>
USRID	客户号	已匿名处理<br>
EVT_LBL	点击模块名称	已清晰并编码<br>
OCC_TIM	触发时间	用户触发该事件的精准时间<br>
TCH_TYP	事件类型	0：APP,1：WEB,2：H5<br>
其中，点击模块名称均为数字编码（形如231-145-18），代表了点击模块的三个级别（如饭票-代金券-门店详情）<br>
#### (3)标注数据：包括客户号及标签。其中，标签数据为用户是否会在未来一周，购买掌上生活APP上的优惠券。具体数据结构如下<br>
字段名	字段含义	说明<br>
USRID	客户号	已匿名处理<br>
FLAG	未来一周是否购买APP上的优惠券	0：未购买，1：购买<br>
测试数据前两部分与训练数据相同，但不提供标注数据<br>
### 6.文件清单和使用说明<br>
train/ ——训练样本目录，包含三个文件<br>
train_agg.csv ——个人属性与信用卡消费数据<br>
train_log.csv ——APP操作行为日志<br>
train_flag.csv ——标注数据<br>
test/ ——评测样本目录，包含两个文件，不提供标注数<br>
test_agg.csv —— 个人属性与信用卡消费数据<br>
test_log.csv—— APP操作行为日志<br>
### 7.提交内容<br>
初赛阶段为csv结果提交：参赛者以csv文件格式，提交test_result.csv评测集预测结果到DF，平台进行在线评分，实时排名<br>
### 8.提交格式<br>
test_result.csv文件格式如下（具体可以参考【提交样例】）<br>
字段名	字段含义<br>
USRID	客户号<br>
RST	预测结果<br>
注意：提交结果必须是[0,1] 之间的小数，结果文件以\t 分割，包含header<br>
### 9.评分方式<br>
AUC(Area under Curve)：Roc曲线下的面积，介于0.1和1之间。Auc作为数值可以直观的评价分类器的好坏，值越大越好。在本赛题中我们选取AUC作为评价指标。现介绍如下<br>
#### (1)混淆矩阵<br>
对于一个二分类问题，混淆矩阵如下：<br>
字段名	预测1	预测0	合计<br>
实际1	TP	FN	TP+FN<br>
实际0	FP	TN	TP+TN<br>
合计	TP+FP	FN+TN	TP+FN+TP+TN<br>
True negative(TN)，称为真阴率，表明实际是负样本预测成负样本的样本数<br>
False positive(FP)，称为假阳率，表明实际是负样本预测成正样本的样本数<br>
False negative(FN)，称为假阴率，表明实际是正样本预测成负样本的样本数<br>
True positive(TP)，称为真阳率，表明实际是正样本预测成正样本的样本数<br>
通过上表我们可以计算一下两个值：<br>
真正类率(true postive rate， TPR)：TPR = TP/(TP+FN)，代表分类器预测的正类中实际正实例占所有正实例的比例<br>
负正类率(false postive rate，FPR)：FPR = FP/(FP+TN)，代表分类器预测的正类中实际负实例占所有负实例的比例<br>
#### (2)ROC曲线：以FPR为横轴，以TPR为纵轴，变回获得ROC曲线<br>
#### (3)AUC也就是上图中蓝色阴影的面积<br>
<br>
# 第二部分：程序<br>
#### 1.先运行xjh_feature.py<br>
#### 2.运行xjh_logistic.py，得到xjh_test_logistic_result.csv<br>
#### 3.运行xjh_lightgbm.py，其中包含交叉验证，注意交叉验证只是对算法进行验证，跟最终实施无关，得到xjh_test_lightgbm_result.csv<br>
#### 4.运行xjh_xgboost.py，得到xjh_test_xgboost_result.csv<br>
#### 5.从算法表现上来看，xgboost=lightgbm>logistic<br>
