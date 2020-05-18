'''用tf-idf做features selection指標'''
import pandas as pd
from itertools import chain
from collections import defaultdict
import re
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from datetime import datetime
from imblearn.over_sampling import SMOTE   #!!!!!
from collections import Counter

month_l = [datetime.strftime(x,'%Y/%m').replace('/0','/') for x in list(pd.date_range(start="2016/1", end="2019/1", freq='M'))]

def preprocess1(classorall, df, class_index, n, stopword): #字詞在每中的tf
    class_TFdict = dict()
    class_DFdict = dict()
    
    for index, row in df.iloc[class_index].iterrows(): 
        text = str(row["title"]) + str(row["content"])  
        text = re.sub(r'[^\w]','', text)  #remove string not unicode decoded
        text = re.sub(r"[A-Za-z0-9]", "", text) #remove english & number
        
        doc_has_word = []
        for t in range(len(text)-(n-1)):
            words = text[t:t+n]  
            # tf, df
            if words not in stopword:
                if words in class_TFdict:      #term freq of that class
                    class_TFdict[words] += 1
                else:
                    class_TFdict[words] = 1
                if words not in doc_has_word:
                    if words in class_DFdict:     #df
                        class_DFdict[words] += 1
                    else:
                        class_DFdict[words] = 1  
                    doc_has_word.append(words)               
    if classorall == 'class':
        key_to_remove = []
        for key, val in class_DFdict.items():  #remove the terms whose df <= 10
            if val <= 2:
                key_to_remove.append(key)
        for key in key_to_remove:
            del class_DFdict[key]
            del class_TFdict[key]    
    return class_TFdict, class_DFdict

def preprocess2(df, class_index, n):  #字詞在每篇文章中的tf
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


# read the files
NewDateOfMonth_df = pd.read_csv('stock_updown_dataset/NewDateOfMonth.csv', encoding="UTF-8")  
NewDateIndex_df = pd.read_csv('stock_updown_dataset/NewDateIndex.csv', encoding="UTF-8")    
#df1 = pd.read_csv('keyword extraction/dataset/utf8/news.csv', encoding="UTF-8")[['post_time','title','content']]
#df2 = pd.read_csv('keyword extraction/dataset/utf8/forum.csv', encoding="UTF-8")[['post_time','title','content']]
#df3 = pd.read_csv('keyword extraction/dataset/utf8/bbs.csv', encoding="UTF-8")[['post_time','title','content']]
#df = pd.concat([df1,df2,df3],axis=0)

firm = 'tsmc'
#keeprow = []
#for index, row in df.iloc[:].iterrows():  
#    text = str(row["title"]) + str(row["content"])   
#    if ('台積電' in text or '張忠謀' in text or '劉得音' in text) and 'NYSE' not in text:      
#        keeprow.append(index)
#df.iloc[keeprow].to_csv('keyword_dataset/{}文章集.csv'.format(firm))     
df = pd.read_csv('keyword_dataset/{}文章集.csv'.format(firm), encoding="UTF-8")   #所有標題或內文含'台積電'的文章     

df[['date','_']]  = df.post_time.str.split(expand=True)  #split a column
df = df.merge(NewDateIndex_df, how = "left", on="date") #outer

ngrm = [2,3,4,5,6] 
stopword_list = []
with open('keyword extraction/dataset/stopwords.txt', 'r', encoding='UTF-8') as file:
    for data in file.readlines():
        data = data.strip()    
        stopword_list.append(data)      

            
for firm in ['tsmc']: #'umc','TAIEX','MediaTek'
    rise_dates = pd.read_csv('keyword_dataset/{}RiseDate.csv'.format(firm), encoding="UTF-8")  #每月選股價上升幅度最高的五天
    down_dates = pd.read_csv('keyword_dataset/{}DownDate.csv'.format(firm), encoding="UTF-8") 
    rise_dates = rise_dates.merge(NewDateIndex_df, how = "left", on="date") #outer
    down_dates = down_dates.merge(NewDateIndex_df, how = "left", on="date") #outer
    stock_df = pd.read_csv('stock_updown_dataset/{}.csv'.format(firm), encoding="UTF-8")             
               
                            
    # 過濾出[每train_nmonth個月]的文章集，算tf&df  
    for train_nmonth in [3,5]:
        predictions_df = pd.DataFrame(columns=('NewDate','# of RFpred==1','# of RFpred==0','# of RFpred==-1','# of SVMpred==1','# of SVMpred==0','# of SVMpred==-1'))         
        
        for remove in ['False','True']:
            for n in range(0,len(month_l)-train_nmonth):  #共三年36個月  #range(0,33) #第32,33,34去預測第35個月
                print('train_iter', n)
                rise_keywords = pd.DataFrame()
                down_keywords = pd.DataFrame()
            
                ''' 過濾出[每三個月]的文章集 '''
                ############################################################
                risedatel = rise_dates[5*n:5*(n+train_nmonth)]['NewDate'] #5:選每月股價上升幅度最高的五天
                risedatel = [r-1 for r in risedatel]
                train_rise = df[pd.DataFrame(df.NewDate.tolist()).isin(risedatel).any(1)] #in any element of risedatel
                downdatel = down_dates[5*n:5*(n+train_nmonth)]['NewDate']
                downdatel = [d-1 for d in downdatel]
                train_down = df[pd.DataFrame(df.NewDate.tolist()).isin(downdatel).any(1)]
                
                y_train = np.concatenate((np.ones(train_rise.shape[0]),np.zeros(train_down.shape[0])))  #看漲文章:1, 看跌文章:0   
                print(y_train.shape)
                test_start = int(NewDateOfMonth_df[n+train_nmonth:n+train_nmonth+1]['NewDateBeginOfMonth'])
                test_end =  int(NewDateOfMonth_df[n+train_nmonth:n+train_nmonth+1]['NewDateEndOfMonth'])  
             
                ''' 分別取出 rise_keywords & down_keywords ，計算 tf-idf '''
                for indexxx, riseordown in enumerate(['看漲','看跌']):   
                    target_doc = train_rise if indexxx==0 else train_down
                    target_doc_count = len(target_doc)
                    sheet1 = pd.DataFrame(columns=('gram','TF_IDF'))             
                    # 聯電
                    class_ngram_count = []   
                    class_ngrm = []  
                    for i in ngrm:    #取出所有 ngram as keywords
                        #print('class ', i)
                        class_TF, class_DF = preprocess1('class', target_doc, list(range(target_doc.shape[0])), i, stopword_list)       
                        class_ngram_count.append(len(class_TF))     
                        # merge (ngram, tf, df)
                        ngram_tf_df = defaultdict(list)
                        for k, v in chain(class_TF.items(), class_DF.items()):
                            ngram_tf_df[k].append(v)
                        class_ngrm.append(list(ngram_tf_df.items()))   
                    # index
                    for i in range(len(ngrm)):
                        for gram,(TF,DF) in class_ngrm[i]:         
                            TF_IDF =(1+math.log10(TF))*math.log10(target_doc_count/DF)                 
                            sheet1.loc[sheet1.shape[0]] = [ item for item in [gram,TF_IDF] ]  # add a row      
                    if indexxx == 0:
                        rise_keywords = sheet1.copy()
                    else:
                        down_keywords = sheet1.copy()
            
            
                '''model training'''
                target_articles = pd.concat([train_rise, train_down],axis=0)
                target_doc_count = target_articles.shape[0]
                topk_index = 'TF_IDF'        # feature selection 
                                           
                if remove == True:
                    r_kw_list = list(rise_keywords['gram'])
                    d_kw_list = list(down_keywords['gram'])
                    for index, row in rise_keywords.iloc[:].iterrows():  
                        if row['gram'] in d_kw_list:
                            rise_keywords = rise_keywords.drop([index],axis=0) #drop row
                    for index, row in down_keywords.iloc[:].iterrows():  
                        if row['gram'] in r_kw_list:
                            down_keywords = down_keywords.drop([index],axis=0) #drop row
                            
                # ### 選top k keywords
                topk = 600  #300,700,...
                rise_topk_keywords = rise_keywords.sort_values(by=[topk_index])[:topk]['gram']
                down_topk_keywords = rise_keywords.sort_values(by=[topk_index])[:topk]['gram']
                topk_keywords = list(rise_topk_keywords)+ list(down_topk_keywords)
                # ### 轉成矩陣: tf-idf
                TFlist = []  # ngram, each document, (keyword, tf)
                DFlist = []  # ngram, (keyword, df)
                for i in ngrm:
                    docs_TF, class_DF = preprocess2(target_articles, list(range(target_doc_count)), i)       
                    TFlist.append(docs_TF)
                    DFlist.append(class_DF)
                # X
                X_train =  np.zeros((target_doc_count, 2*topk)) 
                for i in range(target_doc_count):    # y: 每篇文章 
                    for j, kw in enumerate(topk_keywords): # x: top500 keywords
                        ngrmIndex = len(kw)-2
                        if kw in TFlist[ngrmIndex][i]:
                            X_train[i][j] = (1+math.log10(TFlist[ngrmIndex][i][kw]))*math.log10(target_doc_count/DFlist[ngrmIndex][kw])      
                
        
                # resampling!!!
                X_train_res, y_train_res = SMOTE(k_neighbors=5).fit_resample(X_train, y_train)   #for kneibr in range(1,8):效果差不多
                print(Counter(y_train), Counter(y_train_res))  #{1.0: 923, 0.0: 891} => {1.0: 923, 0.0: 923}
                  
                # normalization!!!                      
                xscaler = preprocessing.StandardScaler().fit(X_train_res)
                X_train_res = xscaler.transform(X_train_res)
        
                # ### training 
                '''RandomForestClassifier'''
                RFmodel = RandomForestClassifier(n_estimators=1000)
                RFmodel.fit(X_train_res, y_train_res)
                '''Support Vector Classifier'''
                SVMmodel = SVC(gamma = 'scale')  
                SVMmodel.fit(X_train_res, y_train_res)
                
        
                '''model testing'''
                for newDate in range(test_start, test_end+1):   #有股價的每一天  #test_stock: each day of april   
                    test_articles = df[ df['NewDate'] == newDate ] 
                    
                    if test_articles.shape[0] == 0:  #當天沒有文章
                        result = [newDate, '--','--','--','--','--','--']
                        predictions_df.loc[predictions_df.shape[0]] = [ item for item in result ]  #add a row
                    else:
                        test_doc = list(range(test_articles.shape[0]))
                        test_doc_count = len(test_doc)
                        # ### 轉成矩陣: tf-idf
                        TFlist = []  # ngram, each document, (keyword, tf)
                        DFlist = []  # ngram, (keyword, df)
                        for i in ngrm:
                            docs_TF, class_DF = preprocess2(test_articles, test_doc, i)       
                            TFlist.append(docs_TF)
                            DFlist.append(class_DF)
                            
                        X_test =  np.zeros((test_doc_count, 2*topk)) 
                        print('X_test',X_test.shape[0])
                        haveFeatures = np.zeros((test_doc_count))                
                        for i in range(test_doc_count):    # y: 每篇文章 
                            for j, kw in enumerate(topk_keywords): # x: top500 keywords
                                ngrmIndx = len(kw)-2
                                if kw in TFlist[ngrmIndx][i]:
                                    X_test[i][j] = (1+math.log10(TFlist[ngrmIndx][i][kw]))*math.log10(test_doc_count/DFlist[ngrmIndx][kw]) 
                                    haveFeatures[i] += 1
                            #print('haveFeatures',haveFeatures[i])      
                            
                        X_test = xscaler.transform(X_test)
                        RFpred = RFmodel.predict(X_test)   
                        SVMpred = SVMmodel.predict(X_test) 
                        for i in range(test_doc_count):     #-1: 該篇文章中沒有字詞出現在 feature space中
                            RFpred[i] = RFpred[i] if haveFeatures[i]>=1 else -1 
                            SVMpred[i] = SVMpred[i] if haveFeatures[i]>=1 else -1
                            
                        result = [newDate, sum(RFpred==1),sum(RFpred==0),sum(RFpred==-1), sum(SVMpred==1),sum(SVMpred==0),sum(SVMpred==-1)]
                        predictions_df.loc[predictions_df.shape[0]] = [ item for item in result ]  #add a row
                    
            predictions_df.to_csv('backtesting_result/backtesting_{}_{}month_{}_{}_v3.csv'.format(firm,train_nmonth,remove,topk))