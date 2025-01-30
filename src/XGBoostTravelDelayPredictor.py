import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import os


class XGBoostTravelDelayPredictor:
    def __init__(self):
        self.model = None

    def __sklearn_tags__(self):
        return super(xgb.XGBRegressor, self)._get_tags()

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, param_grid: dict = None):
        if param_grid is None:
            param_grid = {
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
                "n_estimators": [100, 200, 300],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }

        xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
        grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        verbose=1,
        n_jobs=-1
    )
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        print(f"Najlepsze hiperparametry: {grid_search.best_params_}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
            # Dokonuje predykcji na podstawie modelu XGBoost
            if self.model is None:
                raise ValueError("Model nie został wytrenowany. Najpierw wywołaj metodę train().")

            return self.model.predict(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        y_pred = self.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"R^2 Score: {r2}")

        return mae, rmse, r2
    
    def save_model(self, file_path: str):
        folder_path = "/".join(file_path.split("/")[:-1])
        if folder_path:
            os.makedirs(folder_path, exist_ok=True)

        if self.model is None:
            raise ValueError("Model nie został wytrenowany.")
        
        with open(file_path, "wb") as file:
            pickle.dump(self.model, file)

        print(f"Model został zapisany w lokalizacji: {file_path}")


    def plot_feature_importance(self):
        print("[Status] Generowanie wykresu ważności cech")
        if self.model is None:
            raise ValueError("Model nie został wytrenowany. Najpierw wywołaj metodę train().")

        importance = self.model.feature_importances_
        features = self.model.feature_names_in_

        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette="viridis")
        plt.title("Znaczenie cech")
        plt.xlabel("Znaczenie")
        plt.ylabel("Cechy")
        plt.show()

    def plot_actual_vs_predicted(self, y_test, y_pred):
        print("[Status] Generowanie wykresu rzeczywistych vs. przewidywanych wartości")
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        plt.title("Rzeczywiste vs. Przewidywane")
        plt.xlabel("Rzeczywiste wartości")
        plt.ylabel("Przewidywane wartości")
        plt.show()
