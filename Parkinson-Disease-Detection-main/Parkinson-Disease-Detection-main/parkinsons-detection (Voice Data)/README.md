# parkinsons-detection
Jupyter notebook for screening for Parkinson's Disease based on features from voice recordings. Data downloaded from:
https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/

Compared 5 different machine learning algorithms (logistic regression, K-nearest neighbors, Neural Networks, Random Forest, and XGBoost) to determine which has greater accuracy:

![Confusion Matrices](confusion_matrices.png "Confusion Matrices")

Random forest had the best accuracy, with:

Model accuracy: 95.0,
Precision: 0.94,
Recall: 1.0,
F1: 0.48

Dataset citations:

Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig (2008), 
'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease', 
IEEE Transactions on Biomedical Engineering (to appear).

'Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection', 
Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM. 
BioMedical Engineering OnLine 2007, 6:23 (26 June 2007)
