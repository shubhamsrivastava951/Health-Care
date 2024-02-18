# Classification of Breast-Cancer data

## Table of Content

* Demo
* Overview
* Motivation
* About the data
* Technical Aspect
* Execution
* To Do
* Team

## Demo

## Overview
In this project, I investigate a real dataset i.e. breast-cancer-wisonsin.csv by implementing multiple classification algorithms. First the data is preprocessed by replacing the missing values and normalised using min-max scaler. Then the normalised dataset is trained using multiple classification algorithms such as K-Nearest Neighbour, Logistic Regression, Naïve Bayes, Decision Tree, Support Vector Machine and ensembles - Bagging, Boosting (AdaBoost, Gradient Boosting) and Random Forest. These are further evaluated using the stratified 10-fold cross validation method. A grid search is also applied to find the best parameters for some of the classifiers.

## Motivation
Cancer is among the top few leading causes of death and has eversince proved to be a hurdle in the increased life expectancy across the globe. If identified early, it can be diagnosed with improved prognosis and the chances of survival can increase. Classification and data mining techniques can be used to classify patients into 'Benign' and 'Malignant' groups to avoid unnecessary treatments of patients with benign tumor. To perform classification on real datasets, I felt overwhelmed by the number of possible algorithms to choose from. I read a lot of research papers on 'Classification' in machine learning, and realised that there is no one best-fit-for-all algorithm. There are a lot of performance metrics like training time, accurancy, precision, recall etc to be considered while selecting the best model. Hence, this project aims at comparing the different classification algorithms on the basis of accuracy for forecast modelling and predictive analytics.

## Dataset
The dataset used here is the Breast Cancer Wisconsin which contains 699 examples described by 9 numeric attributes. There are two classes – class1 and class2. The features are computed from a digitized image of a fine needle aspirate of a breast mass of the subject. Benign breast cancer tumours correspond to class1 and malignant breast cancer tumours correspond to class2. This file includes the attribute (feature) headings and each row corresponds to one individual. Missing attributes in the dataset are recorded with a ‘?’.

## Technical Aspect

### Tools and Technologies

1. Python version 3.7.0
2. Numpy: 1.18.2
3. Pandas: 1.0.3
4. Scikit-learn: 0.22.2.post1

### Dataset Preparation

Before applying the classification algorithms, the data was preprocessed in such a way that:
1. The missing attribute values were replaced with the mean value of the column using sklearn.impute.SimpleImputer.
2. Normalisation of each attribute was performed using a min-max scaler to normalise the values between [0,1] with sklearn.preprocessing.MinMaxScaler.
3. The classes class1 and class2 were changed to 0 and 1 respectively.
4. The value of each attribute was formatted to 4 decimal places using .4f.

### Classification Algorithms with 10 fold cross-validation

In this step, multiple classifiers were applied to the pre-processed dataset, in particular: Nearest Neighbor, Logistic Regression, Naïve Bayes, Decision Tree, Bagging, Ada Boost and Gradient Boosting. The performance of these classifiers was evaluated using 10-fold cross validation from
sklearn.model_selection.StratifiedKFold with these options:

cvKFold=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

The function definitions are as follows:

K-Nearest Neighbour : def kNNClassifier(X, y, K)

Logistic Regression : def logregClassifier(X, y)

Naive Bayes : def nbClassifier(X, y)

Decision Tree : def dtClassifier(X, y)

Bagging : def bagDTClassifier(X, y, n_estimators, max_samples, max_depth) 

Adaboost : def adaDTClassifier(X, y, n_estimators, learning_rate, max_depth) 

Gradient Boosting : def gbClassifier(X, y, n_estimators, learning_rate)

### Parameter Tuning
For two other classifiers, Linear SVM and Random Forest, I found the best parameters using grid search with 10-fold stratified cross validation (GridSearchCV in sklearn). The split into training and test subsets should be done using train_test_split from sklearn.model_selection with stratification and random_state=0.

Linear SVM : def bestLinClassifier(X,y)
The grid search considered the following values for the parameters C and gamma: 
C = {0.001, 0.01, 0.1, 1, 10, 100}
gamma = {0.001, 0.01, 0.1, 1, 10, 100}

Random Forest: def bestRFClassifier(X,y)
It use RandomForestClassifier from sklearn.ensemble with information gain and max_features set to ‘sqrt’.
The grid search considered the following values for the parameters n_estimators and max_leaf_nodes: 
n_estimators = {10, 30}
max_leaf_nodes = {4, 16}

These function print the best parameters found, best-cross validation accuracy score and best test set accuracy score.

## Execution

### Input

The program takes 3 command line arguments:
1. The first argument is the path to the data file.
2. The second argument is the name of the algorithm to be executed or the option for print the pre-processed dataset:
  *  NN for Nearest Neighbour.
  *  LR for Logistic Regression.
  *  NB for Naïve Bayes.
  *  DT for Decision Tree.
  *  BAG for Ensemble Bagging DT.
  *  ADA for Ensemble ADA boosting DT.
  *  GB for Ensemble Gradient Boosting.
  *  RF for Random Forest.
  *  SVM for Linear SVM.
  *  P for printing the pre-processed dataset
3. The third argument is optional, and should only be supplied to algorithms which require parameters, namely NN, BAG, ADA and GB. It is the path to the file containing the parameter values for the algorithm. The file is formatted as a csv file like in the following examples:
 For algorithms which do not require any parameters (LR, NB, DT, RF, SVM, P), the third argument should not be supplied.
 
The following examples show how the program would be run:
1. If we want to run the k-Nearest Neighbour classifier, the data is in a file called breast-cancer-wisconsin-normalised.csv, and the parameters are stored in a file called param.csv:

``` python MyClassifier.py breast-cancer-wisconsin-normalised.csv NN param.csv ```

2. We want to run Naïve Bayes and the data is in a file called breast-cancer-wisconsin- normalised.csv.

``` python MyClassifier.py breast-cancer-wisconsin-normalised.csv NB ```

3. We want to run the data pre-processing task and the data is in a file called breast-cancer- wisconsin.csv:

``` python MyClassifier.py breast-cancer-wisconsin.csv P ```
      
 ### Output

1. Suppose the normalised data looks like this:
![GitHub Logo](/logo.png)

The output for option P looks like:

```
0.1343,0.4333,0.5432,0.8589,0.3737,0.9485,0.4834,0.9456,0.4329,0 
0.1345,0.4432,0.4567,0.4323,0.1111,0.3456,0.3213,0.8985,0.3456,1 
0.4948,0.4798,0.2543,0.1876,0.9846,0.3345,0.4567,0.4983,0.2845,0
 
```

2. The accuracies for all the algorithms are as follows:
    
    ```
           run MyClassifier breast-cancer-wisconsin-normalised.csv NN param_NN.csv
           0.9671

           run MyClassifier breast-cancer-wisconsin-normalised.csv LR
           0.9642

           run MyClassifier breast-cancer-wisconsin-normalised.csv NB
           0.9585

           run MyClassifier breast-cancer-wisconsin-normalised.csv DT
           0.9356

           run MyClassifier breast-cancer-wisconsin-normalised.csv BAG param_BAG.csv
           0.9585

           run MyClassifier breast-cancer-wisconsin-normalised.csv ADA param_ADA.csv
           0.9556

           run MyClassifier breast-cancer-wisconsin-normalised.csv GB param_GB.csv
           0.9614

           run MyClassifier breast-cancer-wisconsin-normalised.csv RF
           30
           4
           0.9675
           0.9600

           run MyClassifier breast-cancer-wisconsin-normalised.csv SVM
           1
           0.001
           0.9657
           0.9714
 
 To summarize, SVM is the best model to perform classification for the given dataset as it has the highest accuracy of 97%.
 
 ### To Do
 
 1. Deploy the code
 2. Add diagrams for intuitive visualization
 
### Team
Pooja Mahajan
Sreelaxmi Narayan
 
