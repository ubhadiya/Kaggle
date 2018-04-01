import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,PolynomialFeatures
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV,validation_curve,learning_curve
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv("/Users/kamlesh/Documents/Technical Data/Machine Learning/Kaggle/train.csv")
test_df = pd.read_csv("/Users/kamlesh/Documents/Technical Data/Machine Learning/Kaggle/test.csv")

x_test=test_df;
x_train=train_df;

def fillNullValues(df_null):
    df_null['Age']=df_null['Age'].fillna(df_null['Age'].mean())
    df_null['Embarked']=df_null['Embarked'].fillna(df_null['Embarked'].mode()[0])
    df_null['Fare']=df_null['Fare'].fillna(df_null['Fare'].mean())
    return df_null;

x_train=fillNullValues(x_train);
x_test=fillNullValues(x_test);

x_train['Age']=pd.qcut(x_train['Age'],4,labels=['very_low','low','high','very_high'])
x_train['Fare']=pd.qcut(x_train['Fare'],4,labels=['very_low','low','high','very_high'])
x_test['Age']=pd.qcut(x_test['Age'],4,labels=['very_low','low','high','very_high'])
x_test['Fare']=pd.qcut(x_test['Fare'],4,labels=['very_low','low','high','very_high'])

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', str(name))
    if title_search:
        return title_search.group(1)
    return ""

x_train['Title'] = [get_title(i) for i in x_train['Name']]
x_test['Title'] = [get_title(i) for i in x_test['Name']]

def replaceTitle(df_title):
    title=df_title['Title']
    if title in ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'] :
        return 'Rare'
    elif title == 'Mlle':
        return 'Miss'
    elif title =='Ms':
        return 'Miss'
    elif title=='Mme':
        return 'Mrs'
    else:
        return title

x_train['Title'] = x_train.apply(replaceTitle,axis=1)
x_test['Title'] = x_test.apply(replaceTitle,axis=1)

def addFeature(df_add):
    df_add['FamilySize']=df_add['SibSp'] + df_add['Parch'] + 1;
    df_add['IsAlone']=(df_add['FamilySize']==1).astype(int); 
    return df_add;


x_train=addFeature(x_train);
x_test=addFeature(x_test);

def dropFeature(df_drop):
    drop_elements=['PassengerId','Ticket','Cabin','Name','SibSp','Parch'];
    df_drop=df_drop.drop(drop_elements,axis=1)
    return df_drop


x_train=dropFeature(x_train)
x_test=dropFeature(x_test)



labelencoder_title=LabelEncoder();
labelencoder_sex=LabelEncoder();
labelencoder_embarked=LabelEncoder();
labelencoder_age=LabelEncoder();
labelencoder_fare=LabelEncoder();
x_train['Title']=labelencoder_title.fit_transform(x_train['Title'])
x_train['Sex']=labelencoder_sex.fit_transform(x_train['Sex'])
x_train['Embarked']=labelencoder_embarked.fit_transform(x_train['Embarked'])
x_train['Age']=labelencoder_age.fit_transform(x_train['Age'])
x_train['Fare']=labelencoder_fare.fit_transform(x_train['Fare'])

x_test['Title']=labelencoder_title.fit_transform(x_test['Title'])
x_test['Sex']=labelencoder_sex.transform(x_test['Sex'])
x_test['Embarked']=labelencoder_embarked.transform(x_test['Embarked'])
x_test['Age']=labelencoder_age.transform(x_test['Age'])
x_test['Fare']=labelencoder_fare.transform(x_test['Fare'])
    
#print(x_train[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean());
# print(x_test.head())
# print(x_test.isnull().sum())

y_train=train_df.iloc[:,1];
x_train=x_train.drop('Survived',axis=1);
 
scaler = MinMaxScaler()
x_train_scale=scaler.fit_transform(x_train)
x_test_scale=scaler.fit_transform(x_test)
      
poly = PolynomialFeatures(degree=10)
x_train_poly=poly.fit_transform(x_train_scale)
x_test_poly=poly.fit_transform(x_test_scale)

   

x_train_1, x_train_2, y_train_1, y_train_2 = train_test_split(x_train_scale,y_train, test_size=0.5,random_state=True)


# clf=SGDClassifier()
# clf=SVC(C=1.9,gamma=0.01,degree=6,max_iter=1000)
clf=LogisticRegression(C=1.0)


clf.fit(x_train_1,y_train_1);
print(clf.score(x_train_2,y_train_2))

#============ testing model =============
clf.fit(x_train,y_train);
predicted_test=clf.predict(x_test_poly);
 
submission=pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": predicted_test
    })
 
submission.to_csv('titanic.csv', index=False)
