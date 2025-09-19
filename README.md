# Sales_Forecasting_using_Machine_Learning
This project predicts future sales based on historical data including holidays, promotions, and temperature. It applies data preprocessing, feature engineering, and model comparison (Linear Regression, Random Forest) to select the best forecasting model.

Project Overview:

Sales forecasting is the process of predicting future sales using historical data and external factors.
In this project, we build a predictive model that forecasts sales for a company based on holidays, promotions, and temperature data.

Objectives:

Data Preprocessing → Clean and prepare dataset for analysis.

Feature Engineering → Create additional features (e.g., Day, Month, Weekend, etc.) to improve predictions.

Model Selection → Compare different machine learning models (Linear Regression, Decision Tree, Random Forest).

Model Training → Train the selected models on historical data.

Model Evaluation → Evaluate models using metrics like MAE (Mean Absolute Error) and R² Score.

Model Deployment → Prepare the final model for real-world predictions.

Dataset:

The dataset used is synthetically generated and includes the following columns:

Date → Date of sales

Sales → Sales amount

Holiday → (0 = No holiday, 1 = Holiday)

Temperature → Average temperature of the day

Promotion → (0 = No promotion, 1 = Promotion)

Engineered Features → Year, Month, Day, DayOfWeek, IsWeekend, Sales_Per_Temperature, Promotion_Holiday

Tech Stack:

Python 3.x

Pandas & NumPy → Data manipulation and analysis

Matplotlib & Seaborn → Data visualization

Scikit-learn → Machine Learning models

Statsmodels → Statistical analysis

Model Comparison:

We tested the following models:

Linear Regression

Random Forest Regressor

Each model was evaluated using MAE and R² Score to select the best-performing one.

Future Improvements:

Add more external factors (weather events, special occasions).

Try advanced models (XGBoost, LSTM for time series).

Deploy the model using Flask or Streamlit for real-time forecasting.

How to Run

Clone this repository

git clone https://github.com/arqamzia0900/sales-forecasting.git
cd sales-forecasting

Author: Arqam Zia
