import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
st.set_option('deprecation.showPyplotGlobalUse', False)

data=pd.read_csv("D:/5th Semester/Machine Learning Lab/churn/churn.csv")
data['Churn'].value_counts().plot(kind='bar')
plt.xlabel("Count", labelpad=14)
plt.ylabel("Target Variables", labelpad=14)
plt.title("Count of Target Variables per category")
st.pyplot()

missing = pd.DataFrame((data.isnull().sum())*100/data.shape[0]).reset_index()
plt.figure(figsize=(16,5))
ax= sns.pointplot('index',0,data=missing)
plt.xticks(rotation=90, fontsize=7)
plt.title("Percentage of missing values")
plt.ylabel("Percentage")
st.pyplot()

datas=data.copy()
datas.TotalCharges=pd.to_numeric(datas.TotalCharges, errors='coerce')
datas.loc[datas ['TotalCharges'].isnull() == True]
datas.dropna(how ='any', inplace = True)
labels = ["{0} - {1}".format(i,i+11) for i in range(1, 72,12)]
datas['tenure_group'] = pd.cut(datas.tenure, range(1,75,12), right=False , labels=labels)

datas.drop(columns=['customerID','tenure'], axis=1, inplace=True)
datas['Churn'] = np.where(datas.Churn == 'Yes',1,0)
datas_dummy = pd.get_dummies(datas)
sns.lmplot(data=datas,x='MonthlyCharges',y='TotalCharges', fit_reg=False)
st.pyplot()

Mth = sns.kdeplot(datas_dummy.MonthlyCharges[(datas_dummy["Churn"] == 0) ],
                color="Red", shade = True)
Mth = sns.kdeplot(datas_dummy.MonthlyCharges[(datas_dummy["Churn"] == 1) ],
                ax =Mth, color="Blue", shade= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')
st.pyplot()

Mth = sns.kdeplot(datas_dummy.TotalCharges[(datas_dummy["Churn"] == 0) ],
                color="Red", shade = True)
Mth = sns.kdeplot(datas_dummy.TotalCharges[(datas_dummy["Churn"] == 1) ],
                ax =Mth, color="Blue", shade= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Total Charges')
Mth.set_title('Total charges by churn')
st.pyplot()

plt.figure(figsize=(15,8))
datas_dummy.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')
st.pyplot()

fig, ax = plt.subplots()
sns.heatmap(datas_dummy.corr(), ax=ax)
st.write(fig)

data=pd.read_csv("final_churn.csv")
data=data.drop('Unnamed: 0',axis=1)
x=data.drop('Churn',axis=1)
y=data['Churn']
model_dt=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)
model_dt.fit(x,y)
y_pred=model_dt.predict(x)
st.write('Accuracy of decision tree')
st.write(model_dt.score(x,y))

sm = SMOTEENN()
X_resampled, y_resampled = sm.fit_resample(x,y)
xr_train,xr_test,yr_train,yr_test=train_test_split(X_resampled, y_resampled,test_size=0.2)
model_dt_smote=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)
model_dt_smote.fit(xr_train,yr_train)

yr_predict = model_dt_smote.predict(xr_test)
model_score_r = model_dt_smote.score(xr_test, yr_test)
st.write('Accuracy score of decision tree after resampling')
st.write(model_score_r)
st.write('Confusion matrix')
st.write(metrics.confusion_matrix(yr_test, yr_predict))
from sklearn.ensemble import RandomForestClassifier
model_rf=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)
model_rf.fit(x,y)
y_pred=model_rf.predict(x)
st.write('Accuracy score of Random forest')
st.write(model_rf.score(x,y))

sm = SMOTEENN()
X_resampled1, y_resampled1 = sm.fit_resample(x,y)
xr_train1,xr_test1,yr_train1,yr_test1=train_test_split(X_resampled1, y_resampled1,test_size=0.2)
model_rf_smote=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)
model_rf_smote.fit(xr_train1,yr_train1)
yr_predict1 = model_rf_smote.predict(xr_test1)
model_score_r1 = model_rf_smote.score(xr_test1, yr_test1)
st.write(model_score_r1)
st.write(metrics.confusion_matrix(yr_test1, yr_predict1))

from sklearn.decomposition import PCA
pca = PCA(0.9)
xr_train_pca = pca.fit_transform(xr_train1)
xr_test_pca = pca.transform(xr_test1)
explained_variance = pca.explained_variance_ratio_

model=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)
model.fit(xr_train_pca,yr_train1)
yr_predict_pca = model.predict(xr_test_pca)
model_score_r_pca = model.score(xr_test_pca, yr_test1)
st.write(model_score_r_pca)

from sklearn.preprocessing import LabelEncoder
gender = st.selectbox('Select Gender',('0','1'))
gender = int(gender)
churn = st.selectbox('Select churn', ('Yes', 'No'))
le = LabelEncoder()
churn = le.fit_transform([churn])
st.write(churn)
#prediction = model_score_r1.predict([[gender],[churn],[]])
