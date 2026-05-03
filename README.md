# Simple Linear Regression Model for Salary Prediction

## Overview
This project demonstrates how to build a **Simple Linear Regression (LR)** model to predict salary based on years of experience using a dataset from Kaggle.

The goal is to:
- Load and explore the dataset
- Train a linear regression model
- Evaluate its performance
- Visualize the results

---

## Dataset

- **Source**: Kaggle (Salary Data for Linear Regression)
- **Path (local)**:
    `/Users/raeshadev/.cache/kagglehub/datasets/shubham47/salary-data-dataset-for-linear-regression/versions/1`


- Typical columns:
- `YearsExperience` (Independent Variable)
- `Salary` (Dependent Variable)

---

## Requirements

The following Python libraries are required:

```bash
pip install pandas numpy scikit-learn matplotlib

```
Already installed:

pandas (3.0.2)
numpy (2.4.4)
python-dateutil (2.9.0)
six (1.17.0)

Note: Restart the kernel after installing/updating packages.


---

## Project Structure (Alternate)

salary-lr/
│── data/
│   └── salary_data.csv
│── model/
│   └── train_model.py
│── README.md

---

## Implementation Steps

# 1. Import Libraries

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

# 2. Load Dataset
```
df = pd.read_csv("salary_data.csv")
print(df.head())
```

# 3. Split Data
```
X = df[['YearsExperience']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(    
  X, y, test_size=0.2, random_state=42
)
```
# 4. Train Model
```
model = Simple_LR()
model.fit(X_train, y_train)
```

# 5. Make Predictions
```
y_pred = model.predict(X_test)
```
# 6. Evaluate Model
```
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R2 Score:", r2)
```

# 7. Visualization
```
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
```

## Output


A trained linear regression model


Prediction values for test data


Visualization of regression line


Performance metrics (MSE, R²)




## Key Concept


Linear Regression equation:
```
y = mx + c
```
Where:

y = Salary

x = Years of Experience

m = slope (model coefficient)

c = intercept



## Future Improvements


Add multiple features (Multiple Linear Regression)


Perform feature scaling


Use cross-validation


Deploy using Flask or FastAPI



## License

This project is for educational purposes.
