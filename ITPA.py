
from sklearn.ensemble import IsolationForest
import pandas as pd
from YUtils.OneHotEncode import OneHotForLael
from sklearn.model_selection import train_test_split

# parameters
raw_dataset = './dataset/Localization Data for Person Activity Data Set.txt'
train_file_address = './Result/Activity Recognition/Train(Recognition).csv'
test_file_address = "./Result/Activity Recognition/Test(Recognition).csv"
feature_columns = ['f1','f2','f3']
label_columns = ['label']
target_label = 'label'
SaveTraindata = './Result/Activity Recognition/2Train(Recognition).csv'
SaveTestdata = './Result/Activity Recognition/2Test(Recognition).csv'


def IsolationForest_calulate(train_data_one,test_data):
    # 使用异常检测方法
    clf = IsolationForest()
    # 训练异常检测模型
    clf.fit(train_data_one)
    # 模型预测
    Pre_result = clf.predict(test_data)
    # 计算多少个概率
    prob = len([x for x in Pre_result if x == 1])/len(Pre_result)
    return prob

def load_data():
    # 读取训练集和测试集
    train_content = pd.read_csv(train_file_address)
    test_content = pd.read_csv(test_file_address)
    # 热编码标签
    train_content = OneHotForLael(train_content,target_label,feature_columns)
    test_content = OneHotForLael(test_content,target_label,feature_columns)
    '''
    # 将训练集中的特征与标签分开
    train_features = pd.DataFrame(train_content,columns=feature_columns)
    trian_label = pd.DataFrame(train_content,columns=label_columns)
    # 将测试集中的特征与标签分开
    test_features = pd.DataFrame(test_content,columns=feature_columns)
    test_label = pd.DataFrame(test_content,columns=label_columns)
    return train_features,trian_label,test_features,test_label
    '''
    return train_content,test_content


def DivedByLabel(data,tdata):
    # get the different number of people dataset
    count = data[target_label].value_counts()
    countMap = {}
    for i in count.keys():
        list = pd.DataFrame(data.loc[data[target_label]==i])
        countMap[i] = list

    Tcount = tdata[target_label].value_counts()
    TcountMap = {}
    for n in Tcount.keys():
        Tlist = pd.DataFrame(tdata.loc[tdata[target_label]==n])
        TcountMap[n] = Tlist
    # 返回训练集实例字典、测试集实例字典
    return countMap,TcountMap

import numpy as np
# 参数
EPOCH = 50
SizeOfSpace = 30
SizeOfSubSpace = 10

def GenerateTrainTable():
    # 加载数据
    train_content,test_content = load_data()
    # 得到中间数据
    countMap,TcountMap = DivedByLabel(train_content,test_content)
    # 从训练集中每种类别抽取若干个
    count = train_content[target_label].value_counts()
    # construct a dataframe to store probility for each train instance for each label
    df = pd.DataFrame(columns=count.keys())
    df = pd.concat([df, pd.DataFrame(columns=['label'])],axis =0)
    for epoch in range(EPOCH):
        for w in count.keys():
            # 构建小空间，用于构建训练表
            ConstructTrainData = countMap[w].sample(n= SizeOfSubSpace)
            list = [] # store temp prob
            for i in count.keys():
                traindata = countMap[i].sample(n = SizeOfSpace)
                # 大空间
                Space = traindata.loc[:,feature_columns]
                # 小空间
                SubSpace = ConstructTrainData.loc[:, feature_columns]
                prob = IsolationForest_calulate(Space,SubSpace)
                list.append(prob)

            # 选取小空间内占大多数的类别，作为标签
            w = ConstructTrainData[target_label].value_counts().index[0]
            list.append(w)
            print('list',list)
            from YUtils.util import Add_list_colum
            df = Add_list_colum(list,df)

    # rebbuild index
    df = df.reset_index(drop=True)
    # print(df.head())
    df.to_csv(SaveTraindata)

def GenerateTestTable():
    # 加载数据
    train_content,test_content = load_data()
    # 得到中间数据
    countMap,TcountMap = DivedByLabel(train_content,test_content)
    # 从训练集中每种类别抽取若干个
    count = train_content[target_label].value_counts()
    # construct a dataframe to store probility for each train instance for each label
    df = pd.DataFrame(columns=count.keys())
    df = pd.concat([df, pd.DataFrame(columns=['label'])],axis =0)
    for epoch in range(EPOCH):
        for w in count.keys():
            # 构建小空间，用于构建测试表
            e = np.random.randint(1,len(test_content)/SizeOfSubSpace-1)
            ConstructTestData = test_content[SizeOfSubSpace*e:SizeOfSubSpace*(e+1)]

            list = [] # store temp prob
            for i in count.keys():
                traindata = countMap[i].sample(n = SizeOfSpace)
                # 大空间
                Space = traindata.loc[:,feature_columns]
                # 小空间
                SubSpace = ConstructTestData.loc[:, feature_columns]
                prob = IsolationForest_calulate(Space,SubSpace)
                list.append(prob)

            # 选取小空间内占大多数的类别，作为标签
            w = ConstructTestData[target_label].value_counts().index[0]
            list.append(w)
            print('list',list)
            from YUtils.util import Add_list_colum
            df = Add_list_colum(list,df)

    # rebbuild index
    df = df.reset_index(drop=True)
    print(df.head())
    df.to_csv(SaveTestdata)

def Mytest():

    # ######################################
    train = pd.read_csv(SaveTraindata)
    train_feature = train.iloc[:,0:-1]
    train_label = train.iloc[:,-1]
    test = pd.read_csv(SaveTestdata)
    test_feature = test.iloc[:,0:-1]
    test_label = test.iloc[:,-1]
    #########################################

    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    # fit a CART model to the data
    model = DecisionTreeClassifier()

    # fit a SVM model to the data
    # model = SVC()
    # model = GaussianNB()
    # model = LogisticRegression()

    # fit a k-nearest neighbor model to the data
    import time
    currenttime = time.time()
    # model = KNeighborsClassifier()
    model.fit(train_feature, train_label)
    print(model)
    # make predictions
    expected = test_label
    predicted = model.predict(test_feature)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    # print(metrics.confusion_matrix(expected, predicted))

    result = {}
    result['accuracy'] = metrics.accuracy_score(expected,predicted)
    result['recall'] = metrics.recall_score(expected,predicted,average='micro')
    result['precision'] = metrics.precision_score(expected,predicted,average='micro')
    result['F1-score'] = metrics.f1_score(expected,predicted, average='micro')
    # result['AUC'] = metrics.roc_auc_score(expected,model.decision_function(test_feature))
    result['running time'] = time.time()-currenttime
    print(result)

# Location Data for person activity dataset
def preprocess():
    point_index = pd.read_csv('./dataset/Localization Data for Person Activity Data Set.txt'
                              ,sep=',',iterator= True,chunksize = 1000,header=None)
    # newtable = pd.DataFrame(columns=['f1','f2','f3','label'])
    temp = []
    for i in point_index:
        temp.append(i.iloc[:,4:])
    newtable = pd.concat(temp,axis=0,ignore_index=True)
    newtable.columns = ['f1','f2','f3','label']
    # print(type(newtable))
    print(len(newtable))
    x_train,x_test,y_train,y_test =train_test_split(newtable.iloc[:,0:-1],newtable.iloc[:,-1],test_size=0.25,random_state=0,shuffle=False)
    traindata = pd.concat([x_train,y_train],axis=1,ignore_index=True)
    traindata.columns = ['f1','f2','f3','label']
    testdata = pd.concat([x_test, y_test], axis=1, ignore_index=True)
    testdata.columns = ['f1', 'f2', 'f3', 'label']
    traindata.to_csv('./Result/Localization Data for Person Activity Data Set/Train(LD)(20,30).csv',index=False)
    testdata.to_csv('./Result/Localization Data for Person Activity Data Set/Test(LD)(20,30).csv',index=False)

    print(traindata)

if __name__ == '__main__':
    # GenerateTrainTable()
    # GenerateTestTable()
    Mytest()
    # preprocess()
