import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
df = pd.read_excel("house_data.xlsx")
x = df[['Area_sqft']]
y = df['Price_Lakhs']
model = LinearRegression()
model.fit(x,y)
plt.scatter(x,y,label = 'Actual Data')
plt.plot(x,model.predict(x),label = 'Regression Line')
plt.xlabel("Area(sqft)")
plt.ylabel("Price(in Lakhs)")
plt.title("Simple LinearRegression - House Price Prediction")
plt.legend()
plt.show()
print("---House Price Prediction---")
area = float(input("Enter the area of the house in sqft:"))
predicted_price = model.predict([[area]])[0]
print(F"\n estimated house price for{area}sqft = ₹ {predicted_price:2f}lakhs")
plt.scatter(x,y,label = 'training data')
plt.plot(x,model.predict(x),label = 'RegressionLine')
plt.scatter(area,predicted_price,s = 100,label = 'your input point')
plt.xlabel("area(sqft)")
plt.ylabel("price(in lakhs)")
plt.title("predicted price visualization")
plt.legend()
plt.show()
