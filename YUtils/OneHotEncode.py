

# 针对label 进行one hot编码
from sklearn.preprocessing import LabelEncoder
import pandas as pd



'''
input 原始数据 标签 特征值
return 处理好的数据dataframe
'''
def OneHotForLael(raw,target_label,features):
    # 初始化编码器
    le = LabelEncoder()
    # fit 需要编码的label列
    print(raw[target_label])
    le.fit(raw[target_label])
    # transform label 列
    label = le.transform(raw[target_label])
    # 重新构建label dataframe
    label = pd.DataFrame(label,columns=[target_label])
    # 构建 feature dataframe
    raw = pd.DataFrame(raw,columns=features)
    # 合并 feature label dataframe
    all = pd.concat([raw,label],axis=1,ignore_index=False)
    traindata = pd.DataFrame(all)
    return traindata


def OneHotForFeature(content,column):
    # 对 feature进行处理
    label = pd.get_dummies(content['column'])
    return label
