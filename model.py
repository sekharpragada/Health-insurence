#Importing Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
df = pd.read_csv("new_insurance_data.csv")

# Display dataset preview
print(df.head())
print(df.tail())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Dataset information
print("\nDataset Info:")
print(df.info())


#Data Cleaning

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())


# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype == 'str' or df[col].dtype.name == 'string':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())


# Remove duplicate rows
df.drop_duplicates(inplace=True)


#Exploratory Data Analysis(EDA)

# Outlier Detection using Boxplots
sns.boxplot(y=df['age'])
plt.title("Outliers in Age")
plt.show()

sns.boxplot(y=df['bmi'])
plt.title("Outliers in BMI")
plt.show()

sns.boxplot(y=df['children'])
plt.title("Outliers in Children")
plt.show()

sns.boxplot(y=df['charges'])
plt.title("Outliers in Charges")
plt.show()

# Remove outliers using IQR method
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Convert categorical variables using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Set visualization style
sns.set_theme(style="whitegrid")


# Objective 1: Linear Regression
# To analyze and predict insurance costs using Linear Regression and measure model performance using R² Score.
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Define features and target
x = df.drop(columns=['charges'])
y = df['charges']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.70, random_state=42
)

my_model = LinearRegression()

my_model.fit(x_train, y_train)

y_pred = my_model.predict(x_test)

print("\nPredicted Values:")
print(y_pred)

result = pd.DataFrame(columns=['Actual_value', 'Predicted_value'])
result['Actual_value'] = y_test
result['Predicted_value'] = y_pred

print("\nActual vs Predicted Values:")
print(result)

result = r2_score(y_test, y_pred)
print("\nR2 Score:", result)

mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

rmse = np.sqrt(mse)
print("RMSE:", rmse)

sns.regplot(x=y_pred, y=y_test)
plt.xlabel("Predicted Charges")
plt.ylabel("Actual Charges")
plt.title("Actual vs Predicted Insurance Charges (Linear Regression)")
plt.show()


# Objective 2: KNN Regression
# To predict insurance costs based on customer characteristics using KNN Regression.
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

x = df.drop(columns=['charges'])
y = df['charges']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70, random_state=42)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

my_model = KNeighborsRegressor(n_neighbors=5)
my_model.fit(x_train, y_train)
y_pred = my_model.predict(x_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

sns.regplot(x=y_pred, y=y_test, ci=None)
plt.xlabel("Predicted Charges")
plt.ylabel("Actual Charges")
plt.title("KNN Regression")
plt.show()

knn_score = r2_score(y_test, y_pred)


# Objective 3: Predict Insurance Charges Using Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = df.drop(columns=['charges'])
y = df['charges']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70, random_state=42)

my_model = DecisionTreeRegressor(max_depth=5, random_state=42)
my_model.fit(x_train, y_train)

y_pred = my_model.predict(x_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

sns.scatterplot(x=y_test, y=y_pred, color='teal', alpha=0.7)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Decision Tree Regression")
plt.show()

dt_score = r2_score(y_test, y_pred)

# Objective 4: Predict Insurance Charges Using Random Forest Regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = df.drop(columns=['charges'])
y = df['charges']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70, random_state=42)

my_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
my_model.fit(x_train, y_train)

y_pred = my_model.predict(x_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

sns.scatterplot(x=y_test, y=y_pred, color='darkorange', alpha=0.7)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Random Forest Regression")
plt.show()

rf_score = r2_score(y_test, y_pred)

# Objective 5: Predict Insurance Charges Using Support Vector Regression

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = df.drop(columns=['charges'])
y = df['charges']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70, random_state=42)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

my_model = SVR(kernel='rbf', C=100, epsilon=0.1)
my_model.fit(x_train, y_train)

y_pred = my_model.predict(x_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

sns.scatterplot(x=y_test, y=y_pred, color='slateblue', alpha=0.7)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Support Vector Regression")
plt.show()

svr_score = r2_score(y_test, y_pred)

# Objective 6: Model Comparison

models = ['Linear Regression', 'KNN', 'Decision Tree', 'Random Forest', 'SVR']
scores = [result, knn_score, dt_score, rf_score, svr_score]

sns.barplot(x=models, y=scores, hue=models, palette='Set2', legend=False)
plt.ylabel("R2 Score")
plt.title("Model Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# Objective 7: Best Fit Model

best_index = np.argmax(scores)

print("\nBest Fit Model:", models[best_index])
print("R2 Score:", round(scores[best_index], 4))