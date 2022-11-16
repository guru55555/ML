
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
df = pd.read_csv("Users\ADMIN\Desktop\LP3\diabetes.csv")
df.head()
df.isna().sum()
X = df.drop(["Outcome"], axis=1)
y = df["Outcome"]
X.shape
y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)
Accuracy = (tn+tp)*100/(tp+tn+fp+fn)
print("Accuracy {:0.2f}%:".format(Accuracy))
#Precision 
Precision = tp/(tp+fp)
print("Precision {:0.2f}".format(Precision))
#Recall 
Recall = tp/(tp+fn)
print("Recall {:0.2f}".format(Recall))
#Error rate
err = (fp + fn)/(tp + tn + fn + fp)
print("Error rate {:0.2f}".format(err))
