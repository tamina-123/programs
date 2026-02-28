import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
df = pd.read_excel("customer_churn.xlsx")
X = df[['CustomerAge','MonthlyCharges','Tenure','ContractType','InternetServices','SupportCalls','TotalSpend']]
y = df['Churn']
model = RandomForestClassifier(n_estimators = 100,random_state = 0)
model.fit(X,y)
y_pred = model.predict(X)
accuracy = accuracy_score(y,y_pred)
error = 1 - accuracy
print("Accuracy:",round(accuracy,2))
print("Error Rate:",round(error,2))
print("\n Confusion Matrix:")
print(confusion_matrix(y,y_pred))
print("\n---Customer Churn Prediction---")
age = float(input("CustomerAge:"))
mc = float(input("MonthlyCharges:"))
tenure = float(input("Tenure(months):"))
contract = int(input("ContractType(1 = long,0 = monthly):"))
internet = int(input("InternetServices(1 = Yes,0 = No):"))
calls = int(input("SupportCalls:"))
Spend = float(input("TotalSpend:"))
result = model.predict([[age,mc,tenure,contract,internet,calls,Spend]])
if result[0] == 1:
    print("Customer is likely to CHURN")
else:
    print("Customer is likely to STAY")












