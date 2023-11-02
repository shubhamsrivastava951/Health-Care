import pandas as pd
import numpy as np
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def preprocess_data(dataset):
    breast_cancer = pd.read_csv(dataset,na_values = '?')
    breast_cancer['class'] = breast_cancer['class'].map({'class1': 0, 'class2': 1})
    imputer = SimpleImputer()
    imputer.fit(breast_cancer.values)
    breast_cancer_cleaned = imputer.transform(breast_cancer.values)
    scaler = MinMaxScaler()
    scaler.fit(breast_cancer_cleaned)
    breast_cancer_transformed = pd.DataFrame(scaler.transform(breast_cancer_cleaned), columns=breast_cancer.columns)
    breast_cancer_transformed['class'] = breast_cancer_transformed['class'].astype(int)
    print(breast_cancer_transformed.to_csv(sep=',', index=False,header = False,float_format = '%.4f'), end = "")
    
    
def kNNClassifier(X, y, K):
    cvKFold=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = np.array([])
    for train_index, test_index in cvKFold.split(X,y):
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]
        X_test = X.loc[test_index]
        y_test = y.loc[test_index]
        
        knn = KNeighborsClassifier(K)
        knn.fit(X_train, y_train)
        scores = np.append(scores,knn.score(X_test, y_test))
        
        
    print(np.round(scores.mean(),4), end = "")

def logregClassifier(X, y):
    cvKFold=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = np.array([])
    for train_index, test_index in cvKFold.split(X,y):
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]
        X_test = X.loc[test_index]
        y_test = y.loc[test_index]
        
        lr = LogisticRegression(random_state = 0)
        lr.fit(X_train, y_train)
        scores = np.append(scores,lr.score(X_test, y_test))
        
    print(np.round(scores.mean(),4), end = "")

def nbClassifier(X, y):
    cvKFold=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = np.array([])
    for train_index, test_index in cvKFold.split(X,y):
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]
        X_test = X.loc[test_index]
        y_test = y.loc[test_index]
        
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        scores = np.append(scores,nb.score(X_test, y_test))
        
    print(np.round(scores.mean(),4), end = "")

def dtClassifier(X, y):
    cvKFold=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = np.array([])
    for train_index, test_index in cvKFold.split(X,y):
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]
        X_test = X.loc[test_index]
        y_test = y.loc[test_index]
        
        dt = DecisionTreeClassifier(criterion='entropy',random_state=0)
        dt.fit(X_train, y_train)
        scores = np.append(scores,dt.score(X_test, y_test))
        
    print(np.round(scores.mean(),4), end = "")
    
 
def bagDTClassifier(X, y, n_estimators, max_samples, max_depth):
    cvKFold=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = np.array([])
    for train_index, test_index in cvKFold.split(X,y):
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]
        X_test = X.loc[test_index]
        y_test = y.loc[test_index]
        
        bag_clf = BaggingClassifier(DecisionTreeClassifier(criterion='entropy',max_depth=max_depth,random_state = 0), n_estimators, max_samples, random_state=0)
        bag_clf.fit(X_train, y_train)
        scores = np.append(scores,bag_clf.score(X_test, y_test))
        
    print(np.round(scores.mean(),4), end = "")

def adaDTClassifier(X, y, n_estimators, learning_rate, max_depth):
    cvKFold=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = np.array([])
    for train_index, test_index in cvKFold.split(X,y):
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]
        X_test = X.loc[test_index]
        y_test = y.loc[test_index]
        
        ada_clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy',max_depth=max_depth, random_state=0), n_estimators, random_state=0, learning_rate = learning_rate)
        ada_clf.fit(X_train, y_train)
        scores = np.append(scores,ada_clf.score(X_test, y_test))
        
    print(np.round(scores.mean(),4), end = "")

def gbClassifier(X, y, n_estimators, learning_rate):
    cvKFold=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = np.array([])
    for train_index, test_index in cvKFold.split(X,y):
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]
        X_test = X.loc[test_index]
        y_test = y.loc[test_index]
        
        gb_clf = GradientBoostingClassifier(n_estimators= n_estimators, learning_rate = learning_rate, random_state=0)
        gb_clf.fit(X_train, y_train)
        scores = np.append(scores,gb_clf.score(X_test, y_test))
        
    print(np.round(scores.mean(),4), end = "")

def bestLinClassifier(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0, stratify = y)
    cvKFold=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(SVC(kernel = 'linear', random_state = 0), param_grid,cv=cvKFold.split(X_train,y_train))
    grid.fit(X_train,y_train)
    print(grid.best_params_['C'])
    print(grid.best_params_['gamma'])
    print(np.round(grid.best_score_,4))
    print(np.round(grid.score(X_test, y_test),4))
    
def bestRFClassifier(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0, stratify = y)
    cvKFold=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    param_grid = {'n_estimators':[10, 20, 50, 100], 'max_features':['auto', 'sqrt', 'log2'], 'max_leaf_nodes':[10, 20, 30]}
    grid = GridSearchCV(RandomForestClassifier(criterion = 'entropy', random_state = 0), param_grid,cv=cvKFold.split(X_train,y_train))
    grid.fit(X_train,y_train)
    print(grid.best_params_['n_estimators'])
    print(grid.best_params_['max_features'])
    print(grid.best_params_['max_leaf_nodes'])
    print(np.round(grid.best_score_,4))
    print(np.round(grid.score(X_test, y_test),4))

    
if __name__ == "__main__":
    dataset = sys.argv[1]
    algorithm = sys.argv[2]
    if len(sys.argv) == 4:
        parameter = sys.argv[3]
        param_file = pd.read_csv(parameter)

    if algorithm == 'P':
        preprocess_data(dataset)
        
    
    dataset1 = pd.read_csv(dataset)
    X = dataset1.iloc[:,0:-1]
    y = dataset1.iloc[:,-1]
    
    if algorithm == 'NN':
        kNNClassifier(X,y, param_file.iloc[0,0])
    
    if algorithm == 'LR':
        logregClassifier(X,y)
        
    if algorithm == 'NB':
        nbClassifier(X,y)
        
    if algorithm == 'DT':
        dtClassifier(X,y)
        
    if algorithm == 'BAG':
        bagDTClassifier(X, y, param_file.iloc[0,0], param_file.iloc[0,1], param_file.iloc[0,2])
        
    if algorithm == 'ADA':
        adaDTClassifier(X, y, param_file.iloc[0,0], param_file.iloc[0,1], param_file.iloc[0,2])
        
    if algorithm == 'GB':
        gbClassifier(X, y, param_file.iloc[0,0], param_file.iloc[0,1])
    
    if algorithm == 'SVM':
        bestLinClassifier(X,y)
        
    if algorithm == 'RF':
        bestRFClassifier(X,y)
        
