import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from copy import deepcopy
from statsmodels.stats.weightstats import ztest

def performance_params(cnf_matrix):
    print(cnf_matrix)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    print(TN)
    
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    perfs = np.concatenate((TPR[np.newaxis,:],TNR[np.newaxis,:],PPV[np.newaxis,:],NPV[np.newaxis,:],FPR[np.newaxis,:],FNR[np.newaxis,:],FDR[np.newaxis,:],ACC[np.newaxis,:]), axis=1)
    return perfs

stepdata = "full"

data = np.load('structured_data_' + stepdata + '.npy')

features = [[] for _ in range(data.shape[1])]
# Feature extraction
for trial in data:
    for i, cls in enumerate(trial):
        abs_channel = np.abs(cls)
        res = np.mean(abs_channel, axis=1)
        features[i].append(res)
features = np.array(features)
X_d = np.concatenate(features, axis=0)
n_classes = data.shape[1]
y_classes = np.array([i // (X_d.shape[0] // n_classes) for i in range(X_d.shape[0])])

cv_schem = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

class RFE_pipeline(Pipeline):
    def fit(self, X, y=None, **fit_params):
        """simply extends the pipeline to recover the coefficients (used by RFE) from the last element (the classifier)
        """
        super(RFE_pipeline, self).fit(X, y, **fit_params)
        self.coef_ = self.steps[-1][-1].coef_
        return self

# create a logistic regression classifier
c_MLR = RFE_pipeline([('std_scal',StandardScaler()),('clf',LogisticRegression(C=n_classes,penalty='l2', multi_class='multinomial', solver='lbfgs', max_iter=5000))])

# create a KNN classifier with k=1
c_KNN = KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')

# get coeffs
ind_train, ind_test = next(cv_schem.split(X_d,y_classes))
c_MLR.fit(X_d[ind_train,:], y_classes[ind_train])
coefs = c_MLR.coef_.mean(0).tolist()
max_indexes = []
for x in range(features.shape[2]):
    ag_max = np.argmax(coefs)
    max_indexes.append(ag_max)
    coefs[ag_max] = float("-inf")
# %%
n_rep = 1000
perf = np.zeros([features.shape[2], n_rep, 2])
perf_shuf = np.zeros([features.shape[2], n_rep, 2])
conf_matrix = np.zeros([features.shape[2], n_rep, 2, 3, 3])
shuf_conf_matrix = np.zeros([features.shape[2], n_rep, 2, 3, 3])

RFE_pow = RFE(c_MLR, n_features_to_select=3)
rk_pow = np.zeros([n_rep,data.shape[2]],dtype=np.int32)
fts = []

epsilon = 0.0001
tol = 30
counter = 0
prev_acc = 0
best_acc_i = 0
for num, n_features in enumerate(max_indexes):
    fts.append(n_features)
    X_data = X_d[:,fts]
    acc = []
    print("N_features", num, end="\t")
    for i_rep in range(n_rep):
        if((i_rep+1)%100 == 0):
            print(((1+i_rep)/10), end="%, ")
        ind_train, ind_test = next(cv_schem.split(X_data,y_classes))
        # train and test for original data
        c_MLR.fit(X_data[ind_train,:], y_classes[ind_train])
        if i_rep == 0: 
            c_MLR_original = deepcopy(c_MLR)
            ind_train_original = deepcopy(ind_train)
            ind_test_original = deepcopy(ind_test)
        cscore = c_MLR.score(X_data[ind_test,:], y_classes[ind_test])

        perf[num, i_rep,0] = cscore
        cfmtrx = confusion_matrix(y_true=y_classes[ind_test], y_pred=c_MLR.predict(X_data[ind_test,:]))
        conf_matrix[num, i_rep,0,:,:] += confusion_matrix(y_true=y_classes[ind_test], y_pred=c_MLR.predict(X_data[ind_test,:]))
        
        c_KNN.fit(X_data[ind_train,:], y_classes[ind_train])
        perf[num, i_rep,1] = c_KNN.score(X_data[ind_test,:], y_classes[ind_test])
        conf_matrix[num, i_rep,1,:,:] += confusion_matrix(y_true=y_classes[ind_test], y_pred=c_KNN.predict(X_data[ind_test,:]))  

        # shuffled performance distributions
        shuf_labels = np.random.permutation(y_classes)

        c_MLR.fit(X_data[ind_train,:], shuf_labels[ind_train])
        perf_shuf[num, i_rep,0] = c_MLR.score(X_data[ind_test,:], shuf_labels[ind_test])
        shuf_conf_matrix[num, i_rep,0,:,:] += confusion_matrix(y_true=y_classes[ind_test], y_pred=c_MLR.predict(X_data[ind_test,:]))  

        c_KNN.fit(X_data[ind_train,:], shuf_labels[ind_train])
        perf_shuf[num, i_rep,1] = c_KNN.score(X_data[ind_test,:], shuf_labels[ind_test])
        shuf_conf_matrix[num, i_rep,1,:,:] += confusion_matrix(y_true=y_classes[ind_test], y_pred=c_KNN.predict(X_data[ind_test,:]))  

    counter += 1
    temp = conf_matrix[num, :, 0, :, :].mean(0)
    FN = temp.sum(axis=1) - np.diag(temp)
    TP = np.diag(temp)
    accuracy_c = (TP/(TP+FN)).mean()
    print("\t", accuracy_c, end="")
    if accuracy_c - prev_acc > epsilon:
        counter = 0
    if counter > tol:
        print("im impacient, cutting off at", prev_acc, "index", best_acc_i)
    if accuracy_c > prev_acc:
        prev_acc = accuracy_c
        best_acc_i = num
    print()
#%%
import matplotlib.pyplot as plt
def get_acc(matrix):
    FN = matrix.sum(axis=1) - np.diag(matrix)
    TP = np.diag(temp)
    return (TP/(TP+FN)).mean()


acc_knn = []
acc_mlr = []
acc_mlr_shuff = []
acc_knn_shuff = []
for x in range(1,features.shape[2]):
    print("n features", x)
    print("\tnormal")
    temp = conf_matrix[x, :, 0, :,:].mean(0)
    acc_mlr.append(get_acc(temp))
    temp = conf_matrix[x, :, 1, :,:].mean(0)
    acc_knn.append(get_acc(temp))
    print("\tshuffled")
    temp = shuf_conf_matrix[x, :, 0, :,:].mean(0)
    acc_mlr_shuff.append(get_acc(temp))
    temp = shuf_conf_matrix[x, :, 1, :,:].mean(0)
    acc_knn_shuff.append(get_acc(temp))


plt.plot(acc_mlr, label="MLR")
plt.plot(acc_knn, label="KNN")
plt.plot(acc_mlr_shuff, label="MLR shuffled")
plt.plot(acc_knn_shuff, label="KNN shuffled")
  
# naming the x axis
plt.xlabel('Number of features')
# naming the y axis
plt.ylabel('True positive rate')
  
# giving a title to my graph
plt.title('ML accuracies')

plt.legend()
# function to show the plot
plt.show()
print(np.array(acc_mlr).mean())
print(np.array(acc_knn).mean())

np.save("accuracies_mlr_" + stepdata, np.array(acc_mlr))
np.save("accuracies_knn_" + stepdata, np.array(acc_knn))