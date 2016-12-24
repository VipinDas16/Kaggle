# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 22:39:09 2016   @author:das
"""
import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
path="D:\Analytics\Kaggle\Titanic-redo"
train = pd.read_csv(path+'\\train.csv')
test= pd.read_csv(path+"\\test.csv")

targets = train.Survived
ids=test['PassengerId']
train.drop('Survived',1,inplace=True)
    
 # merging train data and test data for future feature engineering
data = train.append(test)
data.reset_index(inplace=True)
data.drop('index',inplace=True,axis=1)   
    
    
#data['Age'].fillna(data['Age'].median(), inplace=True)
#Processing the titles
def get_titles(data):
    # we extract the title from each name
    data['Title'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())    
    # a map of more aggregated titles
    Title_Dictionary = {"Capt":       "Rare","Col":        "Rare",
                        "Major":      "Rare","Jonkheer":   "Rare",
                        "Don":        "Rare","Sir" :       "Rare",
                        "Dr":         "Rare","Rev":        "Rare",
                        "the Countess":"Rare","Dona":       "Rare",
                        "Mme":        "Mrs","Mlle":       "Miss",
                        "Ms":         "Miss","Mr" :        "Mr",
                        "Mrs" :       "Mrs","Miss" :      "Miss",
                        "Master" :    "Master","Lady" :      "Rare" }    
    # we map each title
    data['Title'] = data.Title.map(Title_Dictionary)
    return data
data=get_titles(data)

#Processing the Age
#Filling median age for Sex--> PClass---> and Title
def process_age(data): 
    #Age imputation by means of Sex and Title
    titles=data.Title.unique()
    for title in titles:
        fillAge=data['Age'] [data['Title']==title].median()
        print(fillAge)
        data.loc[(data['Title'] ==title) & (np.isnan(data['Age'])), 'Age'] = fillAge
    return data    
data=process_age(data)

#Removing Names and keeping only the titles
def process_names(data):    
    # we clean the Name variable
    data.drop('Name',axis=1,inplace=True)    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(data['Title'],prefix='Title')
    data = pd.concat([data,titles_dummies],axis=1)    
    # removing the title variable
    data.drop('Title',axis=1,inplace=True)
    return data
data=process_names(data)
#_________________________________________________________
#handle missing fares
#data.Fare.fillna(data.Fare.mean(),inplace=True)
fillFare=data['Fare'][(data['Pclass']==3) &(data['Embarked']=='S')].median()
data.Fare.fillna(fillFare,inplace=True)
#________________________________________________________
#Processing Embarked
def process_embarked(data):   
    # two missing embarked values - Matches with class median fare
    data.Embarked.fillna('C',inplace=True)   
    # dummy encoding 
    embarked_dummies = pd.get_dummies(data['Embarked'],prefix='Embarked')
    data = pd.concat([data,embarked_dummies],axis=1)
    data.drop('Embarked',axis=1,inplace=True)
    return data    
data=process_embarked(data)
#________________________________________________
#Processing Cabin Variable
def process_cabin(data):    
    # replacing missing cabins with U (for Uknown)
    data.Cabin.fillna('U',inplace=True)    
    # mapping each Cabin value with the cabin letter
    data['Cabin'] = data['Cabin'].map(lambda c : c[0])    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(data['Cabin'],prefix='Cabin')    
    data = pd.concat([data,cabin_dummies],axis=1)    
    data.drop('Cabin',axis=1,inplace=True) 
    return data
data=process_cabin(data)
#______________________________________________________
data['Sex'] = data['Sex'].map({'male':1,'female':0})
#Pclass
def process_pclass(data):
     # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(data['Pclass'],prefix="Pclass")    
    # adding dummy variables
    data = pd.concat([data,pclass_dummies],axis=1)    
    # removing "Pclass"    
    data.drop('Pclass',axis=1,inplace=True)
    return data
data=process_pclass(data)
#___________________________________________________
#Addding new feaures
# introducing a new feature : the size of families (including the passenger)
data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
# introducing other features based on the family size
data['Singleton'] = data['FamilySize'].map(lambda s : 1 if s == 1 else 0)
data['SmallFamily'] = data['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
data['LargeFamily'] = data['FamilySize'].map(lambda s : 1 if 5<=s else 0)
#_____________________________________________________________________

def process_ticket(data):
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip() , ticket)
        ticket = filter(lambda t : not t.isdigit(), ticket)
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'
    # Extracting dummy variables from tickets:
    data['Ticket'] = data['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(data['Ticket'],prefix='Ticket')
    data = pd.concat([data, tickets_dummies],axis=1)
    data.drop('Ticket',inplace=True,axis=1)
    return data
#data=process_ticket(data)
data=data.drop('Ticket',axis=1)

#Scaling the features
feats = list(data.columns)
feats.remove('PassengerId')
data[feats] = data[feats].apply(lambda x: x/x.max(), axis=0)

data.pop('PassengerId')
print ('Features processed and scaled successfully !')
##########################################################
#Modeling
############################################################

def compute_score(clf, X, y,scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5,scoring=scoring)
    return np.mean(xval)

train = data.ix[0:890]
test = data.ix[891:]


#Feature selection
clf = XGBClassifier(50)
model_fit=clf.fit(train, targets)
feat_imp=model_fit.booster().get_fscore()

xgbFeatures=[]
for i,index in enumerate(feat_imp):
    xgbFeatures.append(index)    

featureImp=[]
for feat in xgbFeatures:
    featureImp.append(feat_imp[feat])

features = pd.DataFrame()
features['feature'] = xgbFeatures
features['importance'] = featureImp
features.sort_values(by='importance',ascending=False)

#COnverting into more compact datasets
print("Current train shape:",train.shape)
train_new = train[xgbFeatures]
print("train shape is:",train_new.shape)

print("Current test shapeis ",test.shape)
test_new = test[xgbFeatures]
print("test shape is :",test_new.shape)

model = RandomForestClassifier(50)

parameter_grid = {'max_depth': [1, 2, 3,4,5,6],
                  'n_estimators': [50,100,150,200]}
                  
cross_validation = StratifiedKFold(targets, n_folds=4)#5
grid_search = GridSearchCV(model,param_grid=parameter_grid,
                           cv=cross_validation)
grid_search.fit(train_new, targets)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
#output = grid_search.predict(test_new).astype(int)
output = grid_search.predict(test_new).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = ids
df_output['Survived'] = output
filename='D:/Analytics/Kaggle/Titanic-redo/Titanic_new.54.RF1.csv'
df_output.to_csv(filename, index=False)
print("done")
