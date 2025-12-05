import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# Load dataset
dataset = pd.read_csv('Final_Dataset.csv')
dataset = dataset.drop(columns=['Unnamed: 0'], errors='ignore')

X = dataset.drop(columns=['Production'])
y = dataset['Production']

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Final ExtraTrees model
model = ExtraTreesRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Test R² : {r2:.4f} ({r2*100:.2f}%)")
print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")

# Save model
joblib.dump(model, "model.sav")
print("ExtraTrees model saved as model.sav")
