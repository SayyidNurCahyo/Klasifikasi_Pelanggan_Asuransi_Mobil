#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import sklearn
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn import over_sampling
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,roc_auc_score,roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
import itertools
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit


# In[2]:


from google.colab import files
data=files.upload()


# # Load Dataset Dan Data Understanding 

# In[2]:


data=pd.read_csv("C:/Users/LENOVO/Downloads/archive (7)/train.csv")
data=data.drop('id', axis=1)
data


# In[3]:


data.describe(include="all")


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# # Analisis Eksplorasi Data

# In[6]:


def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal


# In[7]:


pal_vi = get_color('viridis_r', len(data['Previously_Insured'].unique()))
pal_ac = get_color('Accent_r', len(data['Gender'].unique()))
pal_spec = get_color('Spectral', len(data['Vehicle_Age'].unique()))
pal_hsv = get_color('hsv', len(data['Vehicle_Damage'].unique()))
pal_bwr = get_color('bwr', len(data['Response'].unique()))


# In[8]:


fig = px.pie(data, values=data['Gender'].value_counts()[data['Gender'].unique()], names=data['Gender'].unique(),
             color_discrete_sequence=pal_ac)
fig.update_layout(width = 400, height = 300,
                  margin = dict(t=0, l=0, r=0, b=0))
fig.update_traces(textfont_size=12)
fig.show()


# In[9]:


fig = px.histogram(data, x=data['Age'], color_discrete_sequence=pal_ac,color='Gender',barmode='group')
fig.update_layout(width = 600, height = 450,
                  margin = dict(t=0, l=0, r=0, b=0))
fig.update_traces(textfont_size=12)
fig.show()


# In[10]:


fig = px.histogram(data, x=data['Region_Code'], color_discrete_sequence=pal_ac,color='Gender',barmode='group')
fig.update_layout(width = 600, height = 450,
                  margin = dict(t=0, l=0, r=0, b=0))
fig.update_traces(textfont_size=12)
fig.show()


# In[11]:


fig = px.pie(data, values=data['Previously_Insured'].value_counts()[data['Previously_Insured'].unique()], names=data['Previously_Insured'].unique(),
             color_discrete_sequence=pal_vi)
fig.update_traces(textposition='outside', textinfo='percent+label', 
                  hole=.6, hoverinfo="label+percent+name")
fig.update_layout(width = 400, height = 300,
                  margin = dict(t=0, l=0, r=0, b=0))
fig.show()


# In[12]:


fig = px.pie(data, values=data['Vehicle_Age'].value_counts()[data['Vehicle_Age'].unique()], names=data['Vehicle_Age'].unique(),
             color_discrete_sequence=pal_spec)
fig.update_layout(width = 400, height = 300,
                  margin = dict(t=0, l=0, r=0, b=0))
fig.update_traces(textfont_size=12)
fig.show()


# In[13]:


fig = px.pie(data, values=data['Vehicle_Damage'].value_counts()[data['Vehicle_Damage'].unique()], names=data['Vehicle_Damage'].unique(),
             color_discrete_sequence=pal_hsv)
fig.update_layout(width = 400, height = 300,
                  margin = dict(t=0, l=0, r=0, b=0))
fig.update_traces(textfont_size=12)
fig.show()


# In[14]:


data['Annual_Premium'].value_counts()


# In[15]:


a=data[data['Annual_Premium']!=2630.0]
a['Annual_Premium']


# In[16]:


fig = px.histogram(a, x=a['Annual_Premium'], color_discrete_sequence=pal_ac,color='Gender',barmode='group')
fig.update_layout(width = 600, height = 450,
                  margin = dict(t=0, l=0, r=0, b=0))
fig.update_traces(textfont_size=12)
fig.show()


# In[17]:


fig = px.violin(data['Vintage'],box=True)
fig.show()


# In[18]:


sns.set_style('darkgrid')
g = sns.barplot(data=data, x=sorted(data['Response'].unique()), y=data['Response'].value_counts()[sorted(data['Response'].unique())],
                ci=False, palette='gist_rainbow')
g.set_xticklabels(sorted(data['Response'].unique()), rotation=45, fontdict={'fontsize':13})
plt.show()


# In[19]:


fig = px.scatter(data, x='Age', y='Policy_Sales_Channel', color="Response")
fig.show()


# In[20]:


f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(), annot=True, linewidths=0.5, fmt='.5f',ax=ax)
plt.show()


# In[21]:


plt.boxplot(data[['Age','Region_Code','Annual_Premium','Policy_Sales_Channel','Vintage']])
plt.title('Box Plot Data', size=12)
plt.ylabel('Frequency')
plt.show()
# Terlihat bahwa Annual_Premium memiliki outlier yang sangat besar dari IQR data


# # Baseline Model

# In[24]:


labelencoder=LabelEncoder()
data['Gender']=labelencoder.fit_transform(data['Gender'])
data['Vehicle_Age']=labelencoder.fit_transform(data['Vehicle_Age'])
data['Vehicle_Damage']=labelencoder.fit_transform(data['Vehicle_Damage'])
data


# In[25]:


X=data.drop('Response',axis=1)
y=data['Response']


# In[26]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=1,stratify=y)


# In[27]:


model = RandomForestClassifier()
model.fit(X_train,y_train)
pred = model.predict(X_test)


# In[28]:


print(classification_report(y_test,pred))


# # Data Preprocessing

# In[29]:


data=pd.read_csv("train.csv")
data=data.drop('id', axis=1)
data


# In[30]:


def cat(x):
    if x <= 27169:
        return 'Murah'
    elif x > 27169 and x <= 36212:
        return 'Sedang'
    elif x > 36212:
        return 'Mahal'

data['Annual_Cat'] = data['Annual_Premium'].apply(cat)


# In[31]:


def cat(x):
    if x <= 105:
        return 'Baru'
    elif x > 105 and x <= 201:
        return 'Biasa'
    elif x > 201:
        return 'Lama'

data['Vintage_Cat'] = data['Vintage'].apply(cat)


# In[32]:


data.head()


# In[33]:


annual = pd.get_dummies(data['Annual_Cat'])
data=pd.concat([data,annual], axis=1)
data.drop('Annual_Cat', axis=1, inplace=True)

#Drop salah satu variabel hasil encoding
data = data.drop(['Mahal'], axis=1)


# In[34]:


vintage = pd.get_dummies(data['Vintage_Cat'])
data=pd.concat([data,vintage], axis=1)
data.drop('Vintage_Cat', axis=1, inplace=True)

#Drop salah satu variabel hasil encoding
data = data.drop(['Lama'], axis=1)


# In[35]:


#Encoding variabel Gender
labelencoder = LabelEncoder()
data['Gender'] = labelencoder.fit_transform(data['Gender'])


# In[36]:


#Encoding variabel Vehicle_Age

age = pd.get_dummies(data['Vehicle_Age'])
data=pd.concat([data,age], axis=1)
data.drop('Vehicle_Age', axis=1, inplace=True)

#Drop salah satu variabel hasil encoding
data = data.drop(['< 1 Year'], axis=1)


# In[37]:


data.head()


# In[38]:


#Encoding Vehicle Damage

data['Vehicle_Damage'] = labelencoder.fit_transform(data['Vehicle_Damage'])


# In[39]:


Q1 = data['Annual_Premium'].quantile(0.25)
Q3 = data['Annual_Premium'].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data['Annual_Premium'] < (Q1 - 1.5 * IQR)) |(data['Annual_Premium'] > (Q3 + 1.5 * IQR)))]
data


# In[40]:


data['Response'].value_counts('0')


# In[41]:


X_over, y_over = over_sampling.RandomOverSampler().fit_resample(data.drop('Response',axis=1), data['Response'])
data=pd.concat([X_over, y_over], axis=1)
data


# In[42]:


data['Response'].value_counts('0')


# In[43]:


X=data.drop('Response',axis=1)
y=data['Response']


# In[44]:


sc =StandardScaler()
X = sc.fit_transform(X)


# In[45]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=1,stratify=y)


# # Improve Model

# In[46]:


model = RandomForestClassifier()
model.fit(X_train,y_train)
pred = model.predict(X_test)


# In[47]:


print("Classification Report \n")
print(classification_report(y_test,pred))


# In[48]:


print(accuracy_score(y_test,pred))


# # Improve Model Using Hyperparameter Tuning

# In[ ]:


model = RandomForestClassifier()
grid_vals = {'criterion': ['gini', 'entropy', 'log_loss'], 'max_features': ['sqrt', 'log2']}
grid_lr = GridSearchCV(estimator=model, param_grid=grid_vals, scoring='accuracy', 
                       cv=5, refit=True, return_train_score=True) 

#Training and Prediction

grid_lr.fit(X_train, y_train)
preds = grid_lr.best_estimator_.predict(X_test)


# # Metric Evaluation

# In[ ]:


print(accuracy_score(y_test,preds))


# In[ ]:


print(classification_report(y_test,preds))


# In[ ]:


print(confusion_matrix(y_test,preds))


# In[ ]:


print(f1_score(y_test,preds))


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


plt.figure()
plot_confusion_matrix(confusion_matrix(y_test, preds, labels=[0,1]), classes=['Customer Is Interested(0)','Customer Is Not Interested(1)'],normalize= False,
                      title='Confusion matrix')


# In[ ]:



def plot_roc(y_test, y_score):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('RECEIVER OPERATING CHARATERISTIC CURVE')
    plt.legend(loc="lower right")


# In[ ]:


plot_roc(y_test,preds)


# # Cross Validation

# In[49]:


cm=[]
total=[]
ac=[]
se=[]
sp=[]


# In[55]:


ssplit=ShuffleSplit(n_splits=5,test_size=0.20)


# In[56]:


model = RandomForestClassifier()
for train_index, test_index in ssplit.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cm.append((confusion_matrix(y_test, pred)).astype(float))
for j in range (5):
    total.append(sum(sum(cm[j])))
    ac.append((cm[j][0,0]+cm[j][1,1])/total[j])
    se.append(cm[j][0,0]/(cm[j][0,0]+cm[j][0,1]))
    sp.append(cm[j][1,1]/(cm[j][1,0]+cm[j][1,1]))
akurasi=np.mean(ac)
spesifisiti=np.mean(sp)
sensitiviti=np.mean(se)
conf_matrix=confusion_matrix(y_test,pred)
print("confusion matrix = ",conf_matrix)
print("akurasi = ", akurasi)
print("spesifisitas = ", spesifisiti)
print("sensitivitas = ", sensitiviti)
df_k3=pd.DataFrame()
test=dict()
for j in range (3):
    test[j]=[]
for i in range (5):
    test[0].append(ac[i])
    test[1].append(sp[i])
    test[2].append(se[i])
for i in range (3):
    df_k3=pd.concat([df_k3,pd.DataFrame(test[i])],axis=1)
df_k3.columns=['Akurasi','Spesitifitas','Sensitivitas']
df_k3


# # Overfitting Check

# In[57]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=1,stratify=y)
train_scores, test_scores = list(), list()
values = [i for i in range(1, 21)]
for i in values:
    model = RandomForestClassifier(max_depth=i)
    model.fit(X_train, y_train)
    train_yhat = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_yhat)
    train_scores.append(train_acc)
    test_yhat = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_yhat)
    test_scores.append(test_acc)

plt.plot(values, train_scores, '-o', label='Train')
plt.plot(values, test_scores, '-o', label='Test')
plt.legend()
plt.show()


# In[ ]:




