import pandas as pd
import os
from sklearn.model_selection import train_test_split

from src.data_loader import DataLoader
from src.data_visualizer import DataVisualizer
from src.preprocessing import Preprocessor
from src.random_forest_model import RandomForestModel

def main():
    # 1. Wczytanie danych
    data_loader = DataLoader(file_path='./data/ForExportNewApi.csv')
    df = data_loader.load_data()
    print('[Status] Wczytano dane')

    # 2. Preprocessing
    preprocessor = Preprocessor()
    df = preprocessor.data_only_for_country(df, 'POL')
    df_clean = preprocessor.clean_data(df)
    print('[Status] Wyczyszczono dane')
    df_transformed = preprocessor.transform_data(df_clean)
    print('[Status] Przekształcono dane')

    # 3. Wybór zmiennej docelowej
    y = df_transformed['MinsDelay']

    features_to_drop = [
        'MinsDelay',         # kolumna docelowa
        'UpdateTimeUTC', 
        'LocalDateTime', 
        'LocalDateTimeWeekAgo'
    ]

    X = df_transformed.drop(columns=features_to_drop, errors='ignore')
    print('[Status] Przygotowano dane do modelu')

    # 4. Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4.1 Wypisz rozmiary zbiorów treningowego i testowego oraz zapisz je do folderu Analysis jako plik csv
    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'y_test shape: {y_test.shape}')

    # sprawdz czy istnieje folder Analysis, jesli nie to go stworz

    if not os.path.exists('Analysis'):
        os.makedirs('Analysis')

    X_train.to_csv('./Analysis/X_train.csv', index=False)
    X_test.to_csv('./Analysis/X_test.csv', index=False)
    y_train.to_csv('./Analysis/y_train.csv', index=False)
    y_test.to_csv('./Analysis/y_test.csv', index=False)

    # 5. Inicjalizacja i trenowanie modelu
    print('[Status] Rozpoczęto trenowanie modelu...')
    rf_model = RandomForestModel(n_estimators=150, random_state=42)
    rf_model.train(X_train, y_train)
    print('[Status] Wytrenowano model')

    # Wyliczamy przewidywania na zbiorze testowym
    y_pred = rf_model.predict(X_test)

    # 6. Ewaluacja
    mse, r2 = rf_model.evaluate(X_test, y_test)
    print("MSE:", mse)
    print("R2 Score:", r2)

    # 7. Inicjalizacja wizualizatora
    #    Zapisujemy listę kolumn, której model oczekuje:
    model_features = X_train.columns.tolist()

    viz = DataVisualizer(
    model_features=model_features,
    label_encoder_city=preprocessor.label_encoder_city,
    label_encoder_country=preprocessor.label_encoder_country
    )
    
    # 8. Wizualizacje podstawowe
    # 8a. Macierz korelacji na danych treningowych
    viz.plot_correlation_matrix(X_train)

    # 8b. Histogram wybranej kolumny (np. TrafficIndexLive)
    viz.plot_histogram(df_transformed, column='TrafficIndexLive')

    # 8c. Feature Importances
    viz.plot_feature_importances(rf_model.model.feature_importances_, X_train.columns)

    # 8d. Porównanie wartości rzeczywistych i przewidywanych
    viz.plot_prediction_vs_true(y_test, y_pred)

    viz.plot_traffic_by_hour_for_city_and_day(data=df_transformed, city='krakow', day_of_week=0)

    viz.plot_traffic_by_hour_for_city_and_day(data=df_transformed, city='szczecin', day_of_week=0)



if __name__ == "__main__":
    main()
