#!/usr/bin/env python
# coding: utf-8
'''handle Imbalancing Classes'''

import pandas as pd
import re
import math
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE   #!!!!!
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
       
def preprocess(df, class_index, n): 
    class_DFdict = dict()
    docs_TF = []
    
    for index, row in df.iloc[class_index].iterrows():
        text = str(row["title"]) + str(row["content"])  
        text = re.sub(r'[^\w]','', text)  #remove string not unicode decoded
        text = re.sub(r"[A-Za-z0-9]", "", text) #remove english & number

        doc_TFdict = dict()
        doc_has_word = []
        for t in range(len(text)-(n-1)):
            words = text[t:t+n]  
            # tf, df
            #if words not in stopword:
            if words in doc_TFdict:      #term freq of that class
                doc_TFdict[words] += 1
            else:
                doc_TFdict[words] = 1
            if words not in doc_has_word:
                if words in class_DFdict:     #df
                    class_DFdict[words] += 1
                else:
                    class_DFdict[words] = 1  
                doc_has_word.append(words)
        docs_TF.append(doc_TFdict)      
        
    return docs_TF, class_DFdict
    
    
# remove 看漲/看跌關鍵詞集中 重複的關鍵詞 
# dataset_index
# * 3: 標題內容含2個以上關鍵詞 +去重複
# * 4: 關鍵詞in標題+去重複
# 選 top 500 keywords by TF_IDF

firm = 'tsmc' #TAIEX, tsmc, umc, MediaTek
for dataset_index in [4]:  #[3, 4]  #3效果不好~~   #keyword_dataset_CKIP效果不好~~ 
    rise_articles = pd.read_csv('keyword_dataset/{}_{}_rise_articles.csv'.format(dataset_index,firm), encoding="UTF-8")[['title','content']]
    rise_keywords = pd.read_csv('keyword_dataset/{}_{}_rise_keywords.csv'.format(dataset_index,firm), encoding="UTF-8")  
    down_articles = pd.read_csv('keyword_dataset/{}_{}_down_articles.csv'.format(dataset_index,firm), encoding="UTF-8")[['title','content']]
    down_keywords = pd.read_csv('keyword_dataset/{}_{}_down_keywords.csv'.format(dataset_index,firm), encoding="UTF-8")
      
    for remove in [False, True]: 
        if remove == True:
            r_kw_list = list(rise_keywords['gram'])
            d_kw_list = list(down_keywords['gram'])
            #print(rise_keywords.shape, down_keywords.shape)

            for index, row in rise_keywords.iloc[:].iterrows():  
                if row['gram'] in d_kw_list:
                    rise_keywords = rise_keywords.drop([index],axis=0) #drop row
            for index, row in down_keywords.iloc[:].iterrows():  
                if row['gram'] in r_kw_list:
                    down_keywords = down_keywords.drop([index],axis=0) #drop row
        #print(rise_keywords.shape, down_keywords.shape)
      
        for topk in [300,500,700]:   #[300,400,500,600,700]
            for topk_index in ['TF_IDF']:   #['TF_IDF','TF','TF*DF_chisquare','MI','Lift'] 

                # ### 選top 500 keywords
                rise_topk_keywords = rise_keywords.sort_values(by=[topk_index])[:topk]['gram']
                down_topk_keywords = rise_keywords.sort_values(by=[topk_index])[:topk]['gram']
                all_topk_keywords = list(rise_topk_keywords)+ list(down_topk_keywords)
               
                # ### 轉成矩陣
                # x: top500 keywords, y: 每篇文章
                # 內容: tf / tf-idf / tf-df*chisquare
                umc_articles = pd.concat([rise_articles,down_articles],axis=0)
                target_doc = list(range(umc_articles.shape[0]))
                target_doc_count = len(target_doc)
                y = np.concatenate((np.ones(rise_articles.shape[0]),np.zeros(down_articles.shape[0])))  #看漲文章:1, 看跌文章:0 

                # tf, df
                TFlist = []  # ngram, each document, (keyword, tf)
                DFlist = []  # ngram, (keyword, df)
                ngrm = [2,3,4,5,6]
                for i in ngrm:
                    docs_TF, class_DF = preprocess(umc_articles, target_doc, i)       
                    TFlist.append(docs_TF)
                    DFlist.append(class_DF)

                # X
                X =  np.zeros((target_doc_count, 2*topk)) 
                for i in range(target_doc_count):    # y: 每篇文章 
                    for j, kw in enumerate(all_topk_keywords): # x: top500 keywords
                        # tf-idf   
                        # tf: docs_TF[i][kw], df: class_DF[kw]  
                        ngrm = len(kw)-2
                        if kw in TFlist[ngrm][i]:
                            X[i][j] = (1+math.log10(TFlist[ngrm][i][kw]))*math.log10(target_doc_count/DFlist[ngrm][kw])      
                
                            
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                # normalization!!!   
                xscaler = preprocessing.StandardScaler().fit(X_train)
                X_train = xscaler.transform(X_train)
                X_test = xscaler.transform(X_test)
                
                # resampling!!! 兩個classes的文章數大約為3:2
                X_train, y_train = SMOTE(k_neighbors=9).fit_resample(X_train, y_train)   #for kneibr in range(1,8):效果差不多
                # PCA!!! 效果不好~~
                #X_res = PCA(n_components=800).fit_transform(X_res)
                #X_res = MinMaxScaler().fit_transform(X_res)

                # ### training
                '''RandomForestClassifier'''
                n_estimators = [500, 1000, 2000]  #[500, 800, 1000, 2000, 5000]
                for n in n_estimators:
                    model = RandomForestClassifier(n_estimators = n)
                    model.fit(X_train, y_train)
                    ypred = model.predict(X_test)
                    
                    print('RF', remove, dataset_index , topk, topk_index, n)
                    print(confusion_matrix(y_test, ypred))
                    print(classification_report(y_test, ypred))

                '''Support Vector Classifier'''
                for gm in ['scale']: #['scale','auto']表現差不多
                    model = SVC(gamma=gm)  #Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. 'scale':1/(n_features * X.var()), 'auto':1/n_features.
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
    
                    print('SVC', remove, dataset_index , topk, topk_index, gm)
                    print(confusion_matrix(y_test, predictions))
                    print(classification_report(y_test, predictions))
                           
                '''kNN'''
                n_neighbors = [5,9,13]
                for n in n_neighbors:
                    knn = KNeighborsClassifier(n_neighbors=n) #weights='distance'
                    knn.fit(X_train,y_train)
                    pred = knn.predict(X_test)
    
                    print('kNN', remove, dataset_index , topk, topk_index, n)
                    print(confusion_matrix(y_test, pred))
                    print(classification_report(y_test, pred))

                
'''
###聯電### [resampling]
'SVCorRForKNN', remove, dataset_index , topk, topk_index)
# dataset_index=3表現都不好
# topk_index有試了['TF_IDF','TF','TF*DF_chisquare','MI','Lift']，只有TF_IDF預測效果好
 
# f1-score (macro avg) 
RF False 4 [400/500] TF_IDF 1000 0.86   
RF False 4 700 TF_IDF 800 0.86 
RF False 4 [300,600] TF_IDF 1000 0.85  
RF True 4 [300/400/500] TF_IDF 1000 0.82

SVC False 4 [300/700] TF_IDF scale 0.82
SVC True 4 [300/all] TF_IDF scale 0.81
SVC False 4 [400/500/600/700] TF_IDF scale 0.80

kNN False 4 500 TF_IDF 3 0.79
kNN True 4 700 TF_IDF 3 0.79
kNN False 4 [300/600] TF_IDF [3,5] 0.78
kNN False 4 [300/400/500] TF_IDF [5,7,9] 0.77


###台積電### [resampling]
'SVCorRForKNN', remove, dataset_index , topk, topk_index)
# topk_index只有試了['TF_IDF]
    
# f1-score (macro avg) 
RF False 4 [300/all] TF_IDF [500/all] 0.84
 
SVC False 4 500 TF_IDF [scale/auto] 0.84
SVC False 4 [300/all] TF_IDF [scale/auto] 0.83

kNN False 4 500 TF_IDF 9 0.75
kNN False 4 [300/400/500] TF_IDF [5/all] 0.74


###聯發科### [resampling] [PCA效果不好] 
'SVCorRForKNN', remove, dataset_index , topk, topk_index)
# topk_index只有試了['TF_IDF]
    
# f1-score (macro avg) 
RF False 4 500 TF_IDF 1000 0.83
RF False 4 700 TF_IDF [500/2000] 0.83
RF False 4 300 TF_IDF [500/all] 0.82

SVC False 4 300 TF_IDF scale 0.80
    
kNN False 4 700 TF_IDF 5 0.74    


###大盤### [resampling]
'SVCorRForKNN', remove, dataset_index , topk, topk_index)
# topk_index只有試了['TF_IDF]
    
# f1-score (macro avg) 
RF False 4 [500/all] TF_IDF 500 0.87    
RF False 4 [300/500/700] TF_IDF [1000/2000] 0.86
    
SVC False 4 700 TF_IDF [scale/auto] 0.87    
SVC False 4 [300/500] TF_IDF [scale/auto] 0.86
    
kNN False 4 [300/500/700] TF_IDF [5/9/13] 0.78
    
-----------------------------------------------------------------
RF: n_estimators[500, 800, 1000, 2000, 5000]設多少好像預測結果差異很小
SVC gamma[scale/auto] 結果差不多
kNN: n_estimators[5,7,9,11,15,21]設多少好像預測結果差異很小
     weighted = 'uniform' or 'distance'結果差不多
'''