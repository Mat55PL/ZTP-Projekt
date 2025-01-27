# data_visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DataVisualizer:
    """
    Klasa pomocnicza do wizualizacji danych i wyników analizy.
    """
    def __init__(self, model_features, label_encoder_city=None, label_encoder_country=None):
        """
        :param model_features: (opcjonalnie) lista kolumn, jakich model oczekuje.
                            Jeśli None, ustawiamy na pustą listę.
        """
        sns.set_style('whitegrid')
        
        self.model_features = model_features
        self.label_encoder_city = label_encoder_city
        self.label_encoder_country = label_encoder_country

    
    def plot_histogram(self, df: pd.DataFrame, column: str):
        """
        Tworzy histogram dla wybranej kolumny.
        
        :param df: DataFrame z danymi.
        :param column: Nazwa kolumny w df, dla której tworzymy histogram.
        """
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x=column, kde=True)
        plt.title(f'Histogram: {column}')
        plt.show()

    def plot_correlation_matrix(self, df: pd.DataFrame):
        """
        Rysuje macierz korelacji (heatmap).
        
        :param df: DataFrame z danymi numerycznymi.
        """
        plt.figure(figsize=(10, 8))
        correlation = df.corr(numeric_only=True)
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Macierz korelacji')
        plt.show()

    def plot_feature_importances(self, importances, feature_names):
        """
        Rysuje wykres słupkowy z ważnością cech (feature importances) 
        dla modelu, np. Random Forest.
        
        :param importances: Lista lub tablica ważności cech (np. model.feature_importances_).
        :param feature_names: Nazwy cech, w tej samej kolejności co importances.
        """
        # Konwersja na Series (ułatwia sortowanie i rysowanie)
        fi_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=fi_series.values, y=fi_series.index)
        plt.title('Ważność cech (Feature Importances)')
        plt.xlabel('Wartość Feature Importance')
        plt.ylabel('Cecha')
        plt.show()

    def plot_prediction_vs_true(self, y_true, y_pred, title='Porównanie: Rzeczywiste vs. Przewidywane'):
        """
        Rysuje wykres porównujący wartości rzeczywiste (y_true)
        oraz przewidywane (y_pred).
        
        :param y_true: Prawdziwe wartości docelowe (np. DataFrame lub numpy array).
        :param y_pred: Przewidywane wartości modelu (np. DataFrame lub numpy array).
        :param title: Tytuł wykresu.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  # linia odniesienia
        plt.xlabel('Wartości rzeczywiste')
        plt.ylabel('Wartości przewidywane')
        plt.title(title)
        plt.show()

    def plot_traffic_by_hour_for_city_and_day(self, data, city, day_of_week):
        """
        Tworzy wykres natężenia ruchu (TrafficIndexLive) w wybranym mieście i dniu tygodnia.

        Parametry:
            data (pd.DataFrame): Dane wejściowe.
            city (str): Miasto, dla którego chcemy wygenerować wykres.
            day_of_week (int): Numer dnia tygodnia (0 = Poniedziałek, ..., 6 = Niedziela).
        """
        # Jeśli City jest zakodowane, odkoduj je
        if not data['City'].dtype == 'object' and self.label_encoder_city:
            data['City'] = self.label_encoder_city.inverse_transform(data['City'])

        # Dodanie kolumny z dniem tygodnia na podstawie LocalDateTime
        data['LocalDateTime'] = pd.to_datetime(data['LocalDateTime'])
        data['DayOfWeek'] = data['LocalDateTime'].dt.dayofweek

        # Filtrowanie danych dla wybranego miasta i dnia tygodnia
        filtered_data = data[(data['City'].str.lower() == city.lower()) & (data['DayOfWeek'] == day_of_week)]

        if filtered_data.empty:
            print(f"Brak danych dla miasta {city} w dniu {day_of_week}.")
            return

        # Grupowanie danych według godzin
        filtered_data['Hour'] = filtered_data['LocalDateTime'].dt.hour
        hourly_traffic = filtered_data.groupby('Hour')['TrafficIndexLive'].mean()

        # Wykres
        plt.figure(figsize=(10, 6))
        plt.plot(hourly_traffic.index, hourly_traffic.values, marker='o', linestyle='-', label='Traffic Index')
        plt.title(f'Natężenie ruchu w mieście {city.capitalize()} - Dzień tygodnia {day_of_week}')
        plt.xlabel('Godzina')
        plt.ylabel('Natężenie ruchu (TrafficIndexLive)')
        plt.xticks(range(0, 24))
        plt.grid(True)
        plt.legend()
        plt.show()