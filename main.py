import streamlit as st 
import pandas as pd 
import numpy as np 
from nltk.corpus import stopwords 
import string
from sklearn.model_selection import train_test_split,KFold,cross_val_score 
from sklearn.naive_bayes import GaussianNB 
from sklearn import svm 
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier 
from sklearn.preprocessing import LabelEncoder 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import GridSearchCV 
st.set_page_config(page_title = "Web App for Sentiment Analysis")
st.title("Machine Learning Models") 
st.write("---") 
st.write("## Upload a csv file") 
df=st.file_uploader("Upload a csv file", type=".csv") 

if df: 
    df=pd.read_csv(df,encoding='ISO-8859-1') 
    df=df.drop(df.iloc[:,2:],axis=1) 
    df.rename({'v1': 'target', 'v2': 'message'}, axis=1, inplace=True) 
    stopword_list = stopwords.words('english') 
    def clean(text): 
        text = text.translate(str.maketrans('', '', string.punctuation)) 
        text = text.lower() 
        text = [word for word in text.split() if word not in stopword_list] 
        text = ' '.join(text) 
        return text 
    df.message = df.message.apply(clean) 
    st.dataframe(df.head()) 
    label_encoder = LabelEncoder() 
    target = label_encoder.fit_transform(df.target) 
    tfidf = TfidfVectorizer() 
    features=df.iloc[:,-1] 
    # X_train, X_test, y_train, y_test = train_test_split(features,target, test_size = 0.2, random_state = 0) 
    features=tfidf.fit_transform(features).toarray() 
    kfoldv=KFold(5) 

    #NaiveBayesModel
    classifier = GaussianNB() 
    parameters1={'var_smoothing': [1, 0.1, 0.01, 0.001, 0.0001]} 
    classifier1=GridSearchCV(classifier,param_grid=parameters1,scoring='accuracy',cv=kfoldv) 
    classifier1.fit(features,target) 
    nvb_score=classifier1.best_score_ 

    #SVM
    clf = svm.SVC() 
    parameters2={'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf','linear','poly','sigmoid']} 
    clf1=GridSearchCV(clf,param_grid=parameters2,scoring='accuracy',cv=kfoldv) 
    clf1.fit(features,target) 
    svm_score=clf1.best_score_ 

    #Decision Trees 
    dtc = DecisionTreeClassifier() 
    parameters3={'max_depth': range(2,20,4), 'min_samples_leaf': [5, 10, 20, 50, 100], 'criterion': ["gini", "entropy"]} 
    dtc1=GridSearchCV(dtc,param_grid=parameters3,scoring='accuracy',cv=kfoldv) 
    dtc1.fit(features,target) 
    dtc_score=dtc1.best_score_ 

    #XGBoost 
    xgb = XGBClassifier() 
    parameters4={'max_depth':range(2,10,2),'min_child_weight':range(1,6,2)} 
    xgb1=GridSearchCV(xgb,param_grid=parameters4,scoring='accuracy',cv=kfoldv) 
    xgb1.fit(features,target) 
    xgb_score=xgb1.best_score_ 

    model=st.selectbox("Select the model",["Naive Bayes","SVM","Decision Trees", "XGBoost"]) 
    st.set_option('deprecation.showPyplotGlobalUse', False) 
    if model=="Naive Bayes": 
     st.write(nvb_score) 
     # ConfusionMatrixDisplay.from_predictions(y_test,y_pred1) 
     # st.pyplot() 
    elif model=="SVM": 
     st.write(svm_score) 
     # ConfusionMatrixDisplay.from_predictions(y_test,y_pred2) 
     # st.pyplot() 
    elif model=="Decision Trees": 
     st.write(dtc_score) 
     # ConfusionMatrixDisplay.from_predictions(y_test,y_pred3) 
     # st.pyplot() 
    elif model=="XGBoost": 
     st.write(xgb_score) 
     # ConfusionMatrixDisplay.from_predictions(y_test,y_pred4) 
     # st.pyplot() 
 

 