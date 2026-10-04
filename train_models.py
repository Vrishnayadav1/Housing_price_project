import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load dataset
df = pd.read_csv("Housing_Price_Dataset.csv")


# 2. Features and target
X = df[["area", "bedrooms", "bathrooms", "stories", "parking"]]
y = df["price"]


# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Create models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=100,
        random_state=42
    ),
    "SVR": SVR()
}


# 5. Train and compare
results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })


# 6. Display results
results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========\n")
print(results_df.to_string(index=False))


# 7. Save comparison
results_df.to_csv("model_comparison.csv", index=False)

print("\nResults saved to model_comparison.csv")