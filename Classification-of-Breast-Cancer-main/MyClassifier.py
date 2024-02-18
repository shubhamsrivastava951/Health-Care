# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import  cross_val_score



def kNNClassifier(X,y,K):
    X_train, y_train, cvKFold=split1(X,y)
    from sklearn.neighbors import KNeighborsClassifier
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train, y_train)
    scores=cross_val_score(knn, X, y, cv=cvKFold)
    print('{0:.4f}'.format(scores.mean()))
    return(scores,scores.mean())

def logregClassifier(X,y):
    X_train, y_train, cvKFold=split1(X,y)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(random_state = 0).fit(X_train,y_train)
    scores=cross_val_score(lr, X, y, cv=cvKFold)
    print('{0:.4f}'.format(scores.mean()))
    return(scores,scores.mean())

def nbClassifier(X,y):
    X_train, y_train, cvKFold=split1(X,y)
    from sklearn.naive_bayes import GaussianNB
    nb=GaussianNB().fit(X_train ,y_train)
    scores=cross_val_score(nb, X, y, cv=cvKFold)
    print('{0:.4f}'.format(scores.mean()))
    return(scores,scores.mean())

def dtClassifier(X,y):
    X_train, y_train, cvKFold=split1(X,y)
    from sklearn.tree import DecisionTreeClassifier
    dt=DecisionTreeClassifier(criterion='entropy',random_state=0).fit(X_train, y_train)
    scores=cross_val_score(dt, X, y, cv=cvKFold)
    print('{0:.4f}'.format(scores.mean()))
    return(scores,scores.mean())

def bagDTClassifier(X,y,n_estimators, max_samples,max_depth):
    X_train, y_train, cvKFold=split1(X,y)
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    bg = BaggingClassifier(DecisionTreeClassifier(criterion='entropy',random_state=0,max_depth=max_depth),n_estimators=n_estimators,max_samples=max_samples,bootstrap=True,random_state=0 ).fit(X_train, y_train)
    scores=cross_val_score(bg, X, y, cv=cvKFold)
    print('{0:.4f}'.format(scores.mean()))
    return(scores,scores.mean())

def adaDTClassifier(X,y,n_estimators, learning_rate, max_depth):
    X_train, y_train, cvKFold=split1(X,y)
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    ada = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy',max_depth=max_depth,random_state=0),n_estimators=n_estimators,learning_rate=learning_rate,random_state=0 ).fit(X_train, y_train)
    scores=cross_val_score(ada, X, y, cv=cvKFold)
    print('{0:.4f}'.format(scores.mean()))
    return(scores,scores.mean())

def gbClassifier(X,y,n_estimators, learning_rate):
    X_train, y_train, cvKFold=split1(X,y)
    from sklearn.ensemble import GradientBoostingClassifier
    gb = GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=learning_rate,random_state=0 ).fit(X_train, y_train)
    scores=cross_val_score(gb, X, y, cv=cvKFold)
    print('{0:.4f}'.format(scores.mean()))
    return(scores,scores.mean())

def bestRFClassifier(X,y):
    X_train,X_test, y_train, y_test, cvKFold=split2(X,y)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    param_grid={'n_estimators' : [10, 30], 'max_leaf_nodes' : [4, 16]}
    rnd = RandomForestClassifier(criterion='entropy', max_features='sqrt',random_state=0)
    grid_search=GridSearchCV(rnd,param_grid, cv=cvKFold,return_train_score=True)
    grid_search.fit(X_train,y_train)
    best_param=grid_search.best_params_
    print(best_param['n_estimators'])
    print(best_param['max_leaf_nodes'])
    print('{:.4f}'.format(grid_search.best_score_))
    print('{:.4f}'.format(grid_search.score(X_test, y_test)))
    
def bestLinClassifier(X,y):
    X_train,X_test, y_train, y_test, cvKFold=split2(X,y)
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]            }
    lin_svm = SVC(kernel="linear", random_state=0)
    grid_search=GridSearchCV(estimator=lin_svm,param_grid=param_grid, cv=cvKFold,return_train_score=True)
    grid_search.fit(X_train,y_train)
    best_param=grid_search.best_params_
    print(best_param['C'])
    print(best_param['gamma'])
    print('{:.4f}'.format(grid_search.best_score_))
    print('{:.4f}'.format(grid_search.score(X_test, y_test)))
    
         


#splitting function for NN,LR,NB,DT,Ensembles
def split1(X,y):
    #perform Stratified K-fold sampling
    from sklearn.model_selection import StratifiedKFold
    # Create StratifiedKFold object.
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    return X_train, y_train, skf

#splitting function for RF,SVM
def split2(X,y):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test=train_test_split(X,y,stratify=y,random_state=0)
    # Create StratifiedKFold object.
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    return X_train,X_test, y_train, y_test, skf

 
def preprocessing(df):
    # Replacing missing values with mean
    data=df.replace('?',np.nan)
    from sklearn.impute import SimpleImputer
    imputer=SimpleImputer(missing_values=np.nan, strategy='mean')
    cols=data.columns[data.isnull().any()]
    for x in cols:
        data[x] = imputer.fit_transform(data[[x]])
    
    #Convert nominal values of class attribute such that (class 1 => 0, class 2 => 1)
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    for x in data.columns:
        if (data[x].dtype==np.object):
            data[x]=le.fit_transform(data[x])
         
    # Normalization of each attribute using min-max scaler
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler() #creating an object
    scaler.fit(data) #calculate min max value of the training data
    data_norm=scaler.transform(data)
    
    #convert the data which is a numpy array after scaling, back to a dataframe by adding the column names
    data_df=pd.DataFrame(data_norm, columns=data.columns)
         
    #Convert the the value of each attribute such that it should be formatted to 4 decimal places using .4f.
    #pd.options.display.float_format = '{:,.4f}'.format
    data_df=data_df.astype(float).applymap('{:,.4f}'.format)
    
    #Change the format of class variable 
    #Convert nominal values of class attribute such that (class 1 => 0, class 2 => 1)
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    data_df['class']=le.fit_transform(data['class'])
    
    #add column names to the normalized dataset
    data_df.to_csv('testnormalised.csv',index=False, header=list(data.columns))
    
    #print presprocessing output
    x = data_df.to_string(header=False, index=False, index_names=False).split('\n') 
    vals = [','.join(ele.split()) for ele in x]
    vals = '\r\n'.join(vals)
    print(vals)

    #preprocessing end
    
    

def main():
    filename = sys.argv[1]
    df=pd.read_csv(filename)
    
    algo=sys.argv[2]
    
    arguments=len(sys.argv)-1
    
    if arguments==3:
        param=sys.argv[3]
        df_param=pd.read_csv(param)
        #print(param) 
   
    if algo=='P':
        preprocessing(df)
    
    else:
        X=df.iloc[:,:-1]
        y=df.iloc[:,-1]
    
        if algo == 'NN':
            K=df_param['K'].iloc[0]
            kNNClassifier(X,y,K)
        
        elif algo == 'LR':
            logregClassifier(X,y)
        
        elif algo == 'NB':
            nbClassifier(X,y)
        
        elif algo == 'DT':
            dtClassifier(X,y)
        
        elif algo == 'BAG':
            n_estimators=df_param['n_estimators'].iloc[0]
            max_samples=df_param['max_samples'].iloc[0]
            max_depth=df_param['max_depth'].iloc[0]
            bagDTClassifier(X,y,n_estimators, max_samples,max_depth)
            
        elif algo == 'ADA':
            n_estimators=df_param['n_estimators'].iloc[0]
            learning_rate=df_param['learning_rate'].iloc[0]
            max_depth=df_param['max_depth'].iloc[0]
            adaDTClassifier(X,y,n_estimators, learning_rate, max_depth)
            
        elif algo == 'GB':
            n_estimators=df_param['n_estimators'].iloc[0]
            learning_rate=df_param['learning_rate'].iloc[0]
            gbClassifier(X,y,n_estimators, learning_rate)
            
        elif algo == 'RF':
            bestRFClassifier(X,y)
            
        elif algo == 'SVM':
            bestLinClassifier(X,y)
        
        


if __name__=="__main__":
    main()


