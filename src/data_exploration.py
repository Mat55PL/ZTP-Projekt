import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataExplorer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        sns.set_theme(style="whitegrid", font_scale=1.2)

    def print_data_info(self):
        print("Podstawowe informacje o danych:")
        print(self.data.info())
        print("\nPodgląd danych:")
        print(self.data.head())
        print("\nStatystyki opisowe:")
        print(self.data.describe())

    def generate_histogram_for_all_columns(self):
        # Wykresy histogramów dla wszystkich kolumn
        n_cols = len(self.data.columns)
        n_rows = (n_cols + 3) // 4  
        
        plt.figure(figsize=(20, 4*n_rows))
        for i, col in enumerate(self.data.columns, 1):
            plt.subplot(n_rows, 4, i)
            sns.histplot(self.data[col], bins=50, kde=True)
            plt.title(f"Rozkład wartości w kolumnie")
            plt.xlabel(col)
            plt.ylabel("Liczba wystąpień")
        plt.tight_layout()
        plt.show()

    def plot_country_distribution(self, country_column='Country'):
        top_countries = self.data[country_column].value_counts().head(10)
        
        others_percent = (self.data[country_column].value_counts().sum() - top_countries.sum()) / len(self.data) * 100
        
        plt.figure(figsize=(12, 8))
        
        patches, texts, autotexts = plt.pie(
            top_countries,
            labels=top_countries.index,
            autopct='%1.1f%%',
            pctdistance=0.85,
            explode=[0.05] * len(top_countries)
        )

        if others_percent > 0:
            plt.annotate(
                f'Pozostałe: {others_percent:.1f}%',
                xy=(1, 1),
                xytext=(1.2, 1),
            )

        plt.title(f"Top 10 krajów w {country_column}", pad=20)
        

        plt.setp(texts, size=8)
        plt.setp(autotexts, size=8)

        plt.legend(
            patches,
            top_countries.index,
            title="Kraje",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
        
        plt.show()

    def plot_targert_distribution(self, target_column):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[target_column], bins=50, kde=True)
        plt.title(f"Rozkład wartości w kolumnie: {target_column}")
        plt.xlabel(target_column)
        plt.ylabel("Liczba wystąpień")
        plt.show()

    def plot_missing_data(self):
        # Wykres typu heatmap przedstawiający braki w danych
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis')
        plt.title('Brakujące dane')
        plt.show()

    def plot_correlation_matrix(self):
        # Wykres macierzy korelacji
        numeric_data = self.data.select_dtypes(include=["number"])
        
        if numeric_data.empty:
            print("Brak danych numerycznych do obliczenia korelacji.")
            return

        corr_matrix = numeric_data.corr()

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Macierz korelacji")
        plt.show()


    def plot_distribution(self, column: str):
        # Wykres rozkładu wartości w kolumnie
        if column not in self.data.columns:
            raise ValueError(f"Kolumna {column} nie istnieje w danych.")

        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[column], bins=50, kde=True)
        plt.title(f"Rozkład wartości w kolumnie: {column}")
        plt.xlabel(column)
        plt.ylabel("Liczba wystąpień")
        plt.show()

    def plot_top_cities(self, target_column: str, top_n: int = 10):
        # Wykres top miast wg średniej wartości w kolumnie docelowej
        if 'City' not in self.data.columns or target_column not in self.data.columns:
            raise ValueError("Brak wymaganych kolumn 'City' lub docelowej w danych.")

        city_means = self.data.groupby('City')[target_column].mean().sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(12, 6))
        city_means.plot(kind='bar', color='skyblue')
        plt.title(f"Top {top_n} miast wg średniej wartości {target_column}")
        plt.xlabel("Miasto")
        plt.ylabel(f"Średnia {target_column}")
        plt.xticks(rotation=45)
        plt.show()

    def plot_seasonality(self, time_column: str, target_column: str, period: str):
        if time_column not in self.data.columns or target_column not in self.data.columns:
            raise ValueError("Brak wymaganych kolumn czasowych lub docelowych w danych.")

        # Konwersja daty, jeśli nie jest w formacie datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data[time_column]):
            self.data[time_column] = pd.to_datetime(self.data[time_column])

        if period == 'hour':
            self.data['hour'] = self.data[time_column].dt.hour
            group_col = 'hour'
            xlabel = "Godzina dnia"
        elif period == 'day_of_week':
            self.data['day_of_week'] = self.data[time_column].dt.dayofweek
            group_col = 'day_of_week'
            xlabel = "Dzień tygodnia (0 = Poniedziałek, 6 = Niedziela)"
        elif period == 'month':
            self.data['month'] = self.data[time_column].dt.month
            group_col = 'month'
            xlabel = "Miesiąc"
        else:
            raise ValueError("Nieznany okres. Użyj 'hour', 'day_of_week' lub 'month'.")

        plt.figure(figsize=(12, 6))
        sns.boxplot(x=group_col, y=target_column, data=self.data)
        plt.title(f"Rozkład {target_column} w zależności od {xlabel}")
        plt.xlabel(xlabel)
        plt.ylabel(target_column)
        plt.show()

    def plot_missing_summary(self):
        missing_data = self.data.isnull().mean() * 100
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

        if missing_data.empty:
            print("Brak brakujących danych.")
            return

        plt.figure(figsize=(10, 6))
        missing_data.plot(kind='bar', color='salmon')
        plt.title("Procent brakujących danych w kolumnach")
        plt.ylabel("Procent braków (%)")
        plt.xlabel("Kolumny")
        plt.xticks(rotation=45)
        plt.show()