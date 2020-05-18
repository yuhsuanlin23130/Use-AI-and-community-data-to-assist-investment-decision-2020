'''經聯電資料的測試結果，用tf-idf指標去做feature selection所訓練出的模型test acc最好
    所以為了節省大量運算時間，本程式只保留各gram計算 tf-idf 的部分'''
    
import pandas as pd
from itertools import chain
from collections import defaultdict
import re
import math
from openpyxl import Workbook
#import pickle

df1 = pd.read_csv('../dataset/utf8/news.csv', encoding="UTF-8")[['post_time','title','content']]
df2 = pd.read_csv('../dataset/utf8/forum.csv', encoding="UTF-8")[['post_time','title','content']]
df3 = pd.read_csv('../dataset/utf8/bbs.csv', encoding="UTF-8")[['post_time','title','content']]

stopword_list = []
with open('../dataset/stopwords.txt', 'r', encoding='UTF-8') as file:
    for data in file.readlines():
        data = data.strip()    
        stopword_list.append(data)
        
risedown =[['漲','利多','布局','看好','成長','加碼','上升','上看','紅盤','收紅','買超','攀升','高點','高峰','新高','降息','多方','上車','樂觀', '護盤','台股漲'], ['慘綠','跌','下挫','利空', '看衰','減碼','綠盤','低點','收綠','收黑','重挫','賣超','新低','摜破','賠','虧','黑天鵝','空方','停工','信用破產','悲觀','斷頭','損失']]


df = pd.concat([df1,df2,df3],axis=0)
#過濾出屬於目標主題的新聞    
target_doc = [[],[]]
for indexxx, riseordown in enumerate(['看漲','看跌']):     
    for index, row in df.iloc[:].iterrows():  
        text = str(row["title"]) + str(row["content"])   
        title = str(row["title"])
        ########################################
        if ('大盤' in text or '加權指數' in text) and 'NYSE' not in text:      
            #cnt = 0
            for wd in risedown[indexxx]:  
                if wd in title:
                    target_doc[indexxx].append(index) 
                    break
 
# remove the duplicate 
print('# of target docs. ', len(target_doc[0]), len(target_doc[1]))
for item in target_doc[0]:
    if item in target_doc[1]:
        target_doc[0].remove(item)
        target_doc[1].remove(item)
    else:
        None
print('# of target docs. ', len(target_doc[0]), len(target_doc[1]))

for indexxx, riseordown in enumerate(['看漲','看跌']):     
    ###########################################
    saveName = '大盤'+ riseordown 
    df.iloc[target_doc[indexxx]].to_csv(saveName+'文章集.csv')  
    
    
def preprocess(classorall, df, class_index, n, stopword): 
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
            if val <= 10:
                key_to_remove.append(key)
        for key in key_to_remove:
            del class_DFdict[key]
            del class_TFdict[key]
        
    return class_TFdict, class_DFdict


#all_doc_count = df.shape[0]
#all_doc = list(range(all_doc_count)) 

ngrm = [2,3,4,5,6]
# 所有文章
#all_ngram_count = []    
#all_ngrm = [] 
#for i in range(len(ngrm)):
#    all_ngrm.append( pickle.load(open("../all_ngram/all_ngrm["+str(i)+"].pickle", "rb")) )
#    all_ngram_count.append(len(all_ngrm[i]))
    
        
for indexxx, riseordown in enumerate(['看漲','看跌']):   
    ###########################################
    saveName = '大盤'+ riseordown 
    wb = Workbook() 
    sheet1 = wb.create_sheet(saveName) 
    df_col = ['gram','TF','DF','TF_IDF']    
    sheet1.append(df_col) 
    
    target_doc_count = len(target_doc[indexxx])
    # 聯電
    class_ngram_count = []   
    class_ngrm = []  
    for i in ngrm:
        print('class ', i)
        class_TF, class_DF = preprocess('class', df, target_doc[indexxx], i, stopword_list)       
        class_ngram_count.append(len(class_TF))     
        # merge (ngram, tf, df)
        ngram_tf_df = defaultdict(list)
        for k, v in chain(class_TF.items(), class_DF.items()):
            ngram_tf_df[k].append(v)
        class_ngrm.append(list(ngram_tf_df.items()))

    gram_list = [[],[],[],[],[]]  #2gram, 3,4,5,6       
    # 合併可能多餘的子字串 (若彼此gram的DF相等或在誤差內(2%)，則合併gram )
    for i in range(1,len(ngrm)): 
        for gram_1, tf_df_1 in class_ngrm[i]:  
            for gram_0, tf_df_0 in class_ngrm[i-1]: 
                if gram_0 in gram_1 and abs((tf_df_1[1]-tf_df_0[1])/tf_df_0[1])<0.02:
                    class_ngrm[i-1].remove((gram_0, tf_df_0))

    # index
    for i in range(len(ngrm)):
        #all_ngrm_dict = dict(all_ngrm[i])
        for gram,(TF,DF) in class_ngrm[i]:         
            #all_TF = all_ngrm_dict[gram][0]  
            #all_DF = all_ngrm_dict[gram][1]  

            TF_IDF =(1+math.log10(TF))*math.log10(target_doc_count/DF)              
            #all_TF_IDF = (1+math.log10(all_TF))*math.log10(all_doc_count/all_DF)              
            #expected_TF = all_TF/all_doc_count*target_doc_count      
            #expected_DF = all_DF/all_doc_count*target_doc_count 
            #if DF >= expected_DF: 
            #    DF_chisquare = math.pow(DF-expected_DF,2) / expected_DF  
            #else:
            #    DF_chisquare = -1 * (math.pow(DF-expected_DF,2) / expected_DF)
            #MI = DF/(all_DF*target_doc_count)                  
            #Lift = (DF/target_doc_count)/(all_DF/all_doc_count)     #用DF    

            grm = ([gram,TF,DF,TF_IDF])
            sheet1.append(grm)   

    wb.save(saveName + 'keywords.xlsx') 
