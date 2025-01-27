import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def __init__(self):
        self.label_encoder_country = LabelEncoder()
        self.label_encoder_city = LabelEncoder()

    def data_only_for_country(self, df: pd.DataFrame, country: str) -> pd.DataFrame:
        return df[df['Country'] == country]

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:

        print('[Check] Liczba brakujących wartości: \n', df.isnull().sum())
        # usuwanie duplikatów
        df.drop_duplicates(inplace=True)

        # usuwanie brakujących wartości
        df.dropna(inplace=True)

        return df
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # konwersja kolumn z datami na type datetime
        df['UpdateTimeUTC'] = pd.to_datetime(df['UpdateTimeUTC'], errors='coerce') 
        df['LocalDateTimeWeekAgo'] = pd.to_datetime(df['LocalDateTimeWeekAgo'], errors='coerce')

        # konwersja kolumn z datami została dodana w celu ułatwienia analizy danych w dalszych etapach

        # Nowe kolumny czasowe
        df['day_of_week'] = df['UpdateTimeUTC'].dt.dayofweek
        df['hour'] = df['UpdateTimeUTC'].dt.hour

         # Label Encoding dla Country i City
        df['Country'] = self.label_encoder_country.fit_transform(df['Country'])

        self.label_encoder_city.fit(df['City'])
        df['City'] = self.label_encoder_city.transform(df['City'])

        # wypisz mi jak po transformacji wygląda DataFrame dla city i country
        print("Country + City after label encoder:\n", df[['Country', 'City']].head())

        return df
    
    def decode_city(self, encoded_city):
            # Sprawdź, czy miasto istnieje w klasach LabelEncoder
            if encoded_city not in self.label_encoder_city.classes_:
                raise ValueError(f"City '{encoded_city}' is not present in LabelEncoder classes: {self.label_encoder_city.classes_}")
            return encoded_city

    def encode_city(self, city_name):
        # Sprawdź, czy miasto istnieje w klasach LabelEncoder
        if city_name not in self.label_encoder_city.classes_:
            raise ValueError(f"City '{city_name}' is not present in LabelEncoder classes: {self.label_encoder_city.classes_}")
        return self.label_encoder_city.transform([city_name])[0]