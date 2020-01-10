import pandas as pd
# import numpy as np 
# import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


rowData = pd.read_csv('../../Data/breastCancer.csv')

rowData.iloc[:,6] = LabelEncoder().fit_transform(rowData.iloc[:,6].values)

dataset = rowData.iloc[:,1:10]
outcome = rowData.iloc[:,10]


X_train,X_test,Y_train,Y_test = train_test_split(dataset,outcome,random_state=0,test_size=0.4)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# n_neighbors 
# math.sqrt(len(Y_test)) take the real part - 1 = 15 
classifier = KNeighborsClassifier(n_neighbors=15,p=2,metric='euclidean')

classifier.fit(X_train,Y_train)


y_pedect = classifier.predict(X_test)

print('prediction array :')

print(y_pedect)

# #evaluate modale

cm = confusion_matrix(Y_test,y_pedect)

print('=============================================')

print('Confusion Matrix :')
print(cm)

print('=============================================')

print('F1 score :')
print(f1_score(Y_test,y_pedect,pos_label=2))
