import pandas as pd
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate,StratifiedKFold
from sklearn.svm import SVC


X=pd.read_csv("X.csv",sep=' ',header=None, dtype=float)
X=X.values

y=pd.read_csv("y_bush_vs_others.csv", header=None)
y_bush = y.values.ravel()
y=pd.read_csv("y_williams_vs_others.csv", header=None)
y_williams = y.values.ravel()

X.shape

np.sum(y_bush)

np.sum(y_williams)

y_bush.shape

y_williams.shape


knn_1= KNeighborsClassifier(n_neighbors=1,n_jobs=-1)
knn_3= KNeighborsClassifier(n_neighbors=3,n_jobs=-1)
knn_5= KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
svc_best=SVC(C=9.0,kernel ="linear") 
#williams
stratified_cv_results_knn1_william = cross_validate(knn_1 , X, y_williams, cv=StratifiedKFold(n_splits = 3, shuffle=True, random_state = 6657), scoring=('precision', 'recall', 'f1'), return_train_score=False)

stratified_cv_results_knn1_william

stratified_cv_results_knn3_william = cross_validate(knn_3 , X, y_williams, cv=StratifiedKFold(n_splits = 3, shuffle=True, random_state = 6657), scoring=('precision', 'recall', 'f1'), return_train_score=False)

stratified_cv_results_knn3_william

stratified_cv_results_knn5_william = cross_validate(knn_5 , X, y_williams, cv=StratifiedKFold(n_splits = 3, shuffle=True, random_state = 6657), scoring=('precision', 'recall', 'f1'), return_train_score=False)

stratified_cv_results_knn5_william


stratified_cv_results_svc_best_william = cross_validate(svc_best , X, y_williams, cv=StratifiedKFold(n_splits = 3, shuffle=True, random_state = 6657), scoring=('precision', 'recall', 'f1'), return_train_score=False)

stratified_cv_results_svc_best_william

# williams.pickle file generation
pickle.dump((np.mean(stratified_cv_results_knn1_william['test_f1']),np.mean(stratified_cv_results_knn3_william['test_f1']),np.mean(stratified_cv_results_knn5_william['test_f1']),np.mean(stratified_cv_results_svc_best_william['test_f1'])), open('williams.pickle', 'wb'))


# Bush

stratified_cv_results_knn1_bush = cross_validate(knn_1 , X, y_bush, cv=StratifiedKFold(n_splits = 3, shuffle=True, random_state = 6657), scoring=('precision', 'recall', 'f1'), return_train_score=False)

stratified_cv_results_knn1_bush

stratified_cv_results_knn3_bush = cross_validate(knn_3 , X, y_bush, cv=StratifiedKFold(n_splits = 3, shuffle=True, random_state = 6657), scoring=('precision', 'recall', 'f1'), return_train_score=False)

stratified_cv_results_knn3_bush

stratified_cv_results_knn5_bush = cross_validate(knn_5 , X, y_bush, cv=StratifiedKFold(n_splits = 3, shuffle=True, random_state = 6657), scoring=('precision', 'recall', 'f1'), return_train_score=False)

stratified_cv_results_knn5_bush


stratified_cv_results_svc_best_bush = cross_validate(svc_best , X, y_bush, cv=StratifiedKFold(n_splits = 3, shuffle=True, random_state = 6657), scoring=('precision', 'recall', 'f1'), return_train_score=False)

stratified_cv_results_svc_best_bush

# bush.pickle file generation

pickle.dump((np.mean(stratified_cv_results_knn1_bush['test_f1']),np.mean(stratified_cv_results_knn3_bush['test_f1']),np.mean(stratified_cv_results_knn5_bush['test_f1']),np.mean(stratified_cv_results_svc_best_bush['test_f1'])), open('bush.pickle', 'wb'))

