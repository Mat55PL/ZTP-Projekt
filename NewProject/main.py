import pandas as pd
from src.preprocessing import Preprocessor
from src.XGBoostTravelDelayPredictor import XGBoostTravelDelayPredictor
from src.data_exploration import DataExplorer
from src.data_loader import DataLoader
def main():

    # 1. wczytanie danych

    print("[STATUS] Wczytywanie danych...")
    data_path = "./data/ForExportNewApi.csv"
    dataLoader = DataLoader(path=data_path)
    data = dataLoader.load_data()

    # eksploracja danych
    print("[STATUS] Eksploracja danych...")
    explorer = DataExplorer(data)

    explorer.print_data_info()
    # explorer.generate_histogram_for_all_columns()
    explorer.plot_country_distribution()
    explorer.plot_missing_data()
    explorer.plot_correlation_matrix()
    explorer.plot_targert_distribution(target_column='MinsDelay')
    explorer.plot_top_cities(target_column='MinsDelay', top_n=10)


    # 2. przetwarzanie danych
    print("[STATUS] Przetwarzanie danych...")
    preprocessor = Preprocessor(data)

    preprocessor.clear_data()

    preprocessor.encode_categorical_data(columns=["Country", "City"])

    preprocessor.convert_to_datetime('LocalDateTime')
    preprocessor.create_time_features(datetime_column='LocalDateTime')
    preprocessor.preprocess_columns_for_xgboost()
    print("[STATUS] Podział danych na zbiór treningowy i testowy...")
    X_train, X_test, y_train, y_test = preprocessor.split_data(target_column='MinsDelay', test_size=0.2, random_state=42)

    # 3. zapis przetworzonych danych
    print("[STATUS] Zapis przetworzonych danych...")
    preprocessor.save_processed_data(X_train, X_test, y_train, y_test, output_path="./result/")

    data = preprocessor.get_preprocessed_data()

    # 4. trenowanie modelu
    print("[STATUS] Trenowanie modelu XGBoost...")
    predictor = XGBoostTravelDelayPredictor()
    predictor.train(X_train, y_train, param_grid=None)

    # 5. ewaluacja modelu
    print("[STATUS] Ewaluacja modelu...")
    predictor.evaluate(X_test, y_test)

    # 6. zapis modelu
    print("[STATUS] Zapis modelu...")
    predictor.save_model(file_path="./models/xgboost_model.pkl")
    
    # 7. Wizualizacja ważności cech
    print("[STATUS] Wizualizacja ważności cech...")
    predictor.plot_feature_importance()
    predictor.plot_actual_vs_predicted(y_test, predictor.predict(X_test))

if __name__ == "__main__":
    main()