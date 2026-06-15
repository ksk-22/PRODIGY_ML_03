import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

X=[]
y=[]
for label,folder in enumerate(['Cats','Dogs']):
    if not os.path.exists(folder):
        continue
    for file in os.listdir(folder):
        try:
            img=cv2.imread(os.path.join(folder,file),cv2.IMREAD_GRAYSCALE)
            img=cv2.resize(img,(64,64))
            X.append(img.flatten())
            y.append(label)
        except:
            pass

X=np.array(X)
y=np.array(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=SVC(kernel='linear')
model.fit(X_train,y_train)
pred=model.predict(X_test)
print('Accuracy:',accuracy_score(y_test,pred))
print(classification_report(y_test,pred,target_names=['Cat','Dog']))