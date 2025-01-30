from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import os

class Preprocessor:
    def __init__(self, data):
        self.data = data

    def clear_data(self):
        # usuwanie duplikatów
        self.data.drop_duplicates(inplace=True)

        # usuwanie wierszy z brakującymi danymi
        self.data.dropna(inplace=True)

    def encode_categorical_data(self, columns: list):
        encoder = LabelEncoder()

        for column in columns:
            self.data[column] = encoder.fit_transform(self.data[column])

    def convert_to_datetime(self, column):
        self.data[column] = pd.to_datetime(self.data[column])

    def create_time_features(self, datetime_column: str):
        if datetime_column not in self.data.columns:
            raise ValueError(f"Kolumna {datetime_column} nie istnieje w danych.")
        
        self.data[datetime_column] = pd.to_datetime(self.data[datetime_column])
        self.data['hour'] = self.data[datetime_column].dt.hour
        self.data['day_of_week'] = self.data[datetime_column].dt.dayofweek
        self.data['is_weekend'] = self.data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        self.data['month'] = self.data[datetime_column].dt.month

    def split_data(self, target_column: str, test_size: float = 0.2, random_state: int = 42):
        # podział danych na zbiór treningowy i testowy 
        if target_column not in self.data.columns:
            raise ValueError(f"Kolumna {target_column} nie istnieje w danych.")
        
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, output_path: str = "./"):
        if output_path[-1] != "/":
            output_path += "/"

        output_path += pd.Timestamp.now().strftime("%Y%m%d_%H%M%S") + "/"

        os.makedirs(output_path, exist_ok=True)

        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        train_data.to_csv(output_path + "train_data.csv", index=False)
        test_data.to_csv(output_path + "test_data.csv", index=False)

        print(f"Dane zostały zapisane w lokalizacji: {output_path}")

    def preprocess_columns_for_xgboost(self):
        # Convert datetime columns to numeric features
        self.data['LocalDateTime'] = pd.to_datetime(self.data['LocalDateTime']).astype(int) 
        self.data['LocalDateTimeWeekAgo'] = pd.to_datetime(self.data['LocalDateTimeWeekAgo']).astype(int) 
        self.data['UpdateTimeUTC'] = pd.to_datetime(self.data['UpdateTimeUTC']).astype(int) 

    def get_preprocessed_data(self):
        return self.data