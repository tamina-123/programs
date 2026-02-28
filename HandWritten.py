import numpy as np 
import cv2
import matplotlib.pyplot as plt 
from sklearn.dataset import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
digits = load_digits()
X = digits.data
y = digits.target
X_train,X_test,y_train,y_test = train_test_split(X,ytest_size = 0.2,random_state = 42)
model = KNeighborsClassifier(n_neighbors = 3)







