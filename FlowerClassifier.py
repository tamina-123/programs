import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
iris = load_iris()
df = pd.DataFrame(iris.data,columns = iris.feature_names)
df['Species'] = iris.target
df['Species'] = df['Species'].replace({0:'setosa',1:'versicolor',2:'virginica'})
print("Sample Date:")
print(df.head())
X = df[iris.feature_names]
y = df['Species']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 1)
model = DecisionTreeClassifier(criterion = 'entropy',random_state = 1)
model.fit (X_train,y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print(f"\n Model Accuracy:{acc*100:.2f}%")
plt.figure(figsize = (12,8))
plot_tree(model,filled = True,feature_names =iris.feature_names,class_names = iris.target_names)
plt.title("Decision Tree for Iris Flower Classification")
plt.show()
print("\n---Predict Flower Type---")
Sepal_length = float(input("Enter Sepal Length(cm):"))
Sepal_width = float(input("Enter sepal Width(cm):"))
Petal_length = float(input("Enter Petal Length(cm):"))
Petal_width = float(input("Enter Petal Width(cm):"))
input_data = [[Sepal_length,Sepal_width,Petal_length,Petal_width]]
prediction = model.predict(input_data)[0]
print(f"\n The Predicted Flower Species is: {prediction.capitalize()}")












