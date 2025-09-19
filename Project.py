import pandas as pd
import numpy as np

# Generate Random Data and Data Preprocessing:

rows = 1100

np.random.seed(42)

dates = pd.date_range(start= '2022-01-01', periods=rows, freq='D')
# print(dates)

sales = np.random.randint(1000, 50000, size=rows)
# print(sales)

holiday = np.random.choice([0,1], size=rows, p=[0.85, 0.15])
# print(holiday)

temperature = np.random.uniform(10, 40, size=rows).round(1)
# print(temperature)

promotion = np.random.choice([0,1], size=rows, p=[0.70, 0.30])
# print(promotion)

frame = pd.DataFrame({
    'Date' : dates,
    'Sales' : sales,
    'Holiday' : holiday,
    'Temperature' : temperature,
    'Promotion' : promotion
})

frame = frame.sort_values('Date').reset_index(drop=True)

# Feature Engineering:

frame['Year'] = frame['Date'].dt.year
frame['Month'] = frame['Date'].dt.month
frame['Day'] = frame['Date'].dt.day
frame['DayOfWeek'] = frame['Date'].dt.dayofweek
frame['IsWeekEnd'] = frame['DayOfWeek'].isin([5,6]).astype(int)

frame['Sales_Per_Temperature'] = (frame['Sales'] / frame['Temperature']).round(2)
frame['Promotion_Holiday'] = (frame['Promotion'] * frame['Holiday'])


frame.to_csv('Sales_Forcasting_Data.csv', index=False)

frame = pd.read_csv('Sales_Forcasting_Data.csv')

frame['Date'] = pd.to_datetime(frame['Date'])

# print(frame.head())
# print(frame.info())

import matplotlib.pyplot as plt


plt.boxplot(frame['Sales'])
plt.title("Sales Outliers")        # Sales ke outliers check
# plt.show()

plt.boxplot(frame['Temperature'])
plt.title("Temperature Outliers")     # Temperature ke outliers check
# plt.show()


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Scale karna sirf numeric columns ko

frame[['Sales', 'Temperature']] = scaler.fit_transform(frame[['Sales', 'Temperature']])

# print(frame[['Sales', 'Temperature']].head())
# print(frame)

# Model Training:

from sklearn.model_selection import train_test_split

x = frame[['Holiday', 'Temperature', 'Promotion', 'Year', 'Month', 'Day', 'DayOfWeek', 'IsWeekEnd', 'Sales_Per_Temperature', 'Promotion_Holiday']]
y = frame['Sales']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model Selection:

# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# model_1 = LinearRegression()
model_2 = RandomForestRegressor(random_state=42, n_estimators=100)

model_2.fit(x_train, y_train)

y_predict = model_2.predict(x_test)

# Model Evaluation:

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_predict) 
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('Model average kitna galat predict kar raha hai: ', mae)
print('Agar model mein zaida error howe to yeh hame bata de ga: ', mse)
print('Overall accuracy batata ha: ',r2 )