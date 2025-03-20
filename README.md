# Projekt ZTP - System prognozowania opóźnień podróży 🚗📊


## 🎯 Cel Projektu

Celem projektu jest stworzenie zaawansowanego systemu przewidywania opóźnień w ruchu drogowym na podstawie danych historycznych i bieżących. System wykorzystuje techniki eksploracji danych, przetwarzania danych oraz algorytmy uczenia maszynowego (XGBoost).

---

## 🗂️ Opis Danych

Projekt wykorzystuje zbiór danych zawierający **182 493 rekordów** oraz **13 kolumn**, opisujących m.in.:

- Kraj i miasto
- Czas aktualizacji (UTC i lokalny)
- Opóźnienia spowodowane korkami (`JamsDelay`)
- Wskaźniki natężenia ruchu (aktualne i sprzed tygodnia)
- Długość i liczba korków
- Czas podróży na 10 km (rzeczywisty i historyczny)
- Opóźnienia w minutach (`MinsDelay`)

---

## 🔍 Etapy projektu

Projekt obejmuje:

1. **Analizę Eksploracyjną Danych (EDA)** - Wizualizacja i eksploracja wzorców ruchu.
2. **Preprocessing danych** - Czyszczenie, kodowanie zmiennych oraz ekstrakcja cech czasowych.
3. **Budowanie modelu XGBoost** - Trening oraz optymalizacja modelu do przewidywania opóźnień.
4. **Ewaluację** - Analiza wyników predykcji i ocena jakości modelu.

---

## 🛠️ Preprocessing danych

Kluczowe działania to:

- Usuwanie duplikatów oraz braków danych.
- Kodowanie zmiennych kategorycznych (Label Encoding).
- Konwersja zmiennych czasowych oraz tworzenie nowych cech czasowych (`hour`, `day_of_week`, `month`, `is_weekend`).
- Podział danych na zestaw treningowy (80%) i testowy (20%).

---

## 🌳 Model XGBoost

**Dlaczego XGBoost?**

- Szybkość działania.
- Wysoka dokładność predykcji.
- Efektywność w obsłudze dużych zbiorów danych.
- Możliwość interpretacji znaczenia cech.

**Hiperparametry użyte w modelu (GridSearchCV):**

| Hiperparametr      | Wartości testowane       |
|--------------------|--------------------------|
| `max_depth`        | `[3, 5, 7]`              |
| `learning_rate`    | `[0.01, 0.1, 0.3]`       |
| `n_estimators`     | `[100, 200, 300]`        |
| `subsample`        | `[0.8, 1.0]`             |
| `colsample_bytree` | `[0.8, 1.0]`             |

---

## 📈 Ewaluacja modelu

Model osiągnął bardzo dobre wyniki:

- **Mean Absolute Error (MAE):** `0.065`
- **Root Mean Squared Error (RMSE):** `0.111`
- **R² Score:** `0.995`

**Najważniejsze cechy:**

1. `TrafficIndexLive` (aktualne natężenie ruchu)
2. `TrafficIndexWeekAgo` (natężenie ruchu sprzed tygodnia)
3. `TravelTimeLivePer10KmsMins` (rzeczywisty czas przejazdu na 10 km)

---

## 📊 Wizualizacje

W projekcie wykonano wizualizacje takie jak:

- Histogramy rozkładu danych
- Macierz korelacji
- Porównanie predykcji z wartościami rzeczywistymi

---

## 📚 Bibliografia i technologie

Projekt wykorzystuje następujące technologie oraz dokumentacje:

- [XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- [Matplotlib](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html)
- [Scikit-Learn GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Scikit-Learn LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
- [Tomtom Data](https://www.kaggle.com/datasets/bwandowando/tomtom-traffic-data-55-countries-387-cities/data)

---

## 💡 Możliwe zastosowania systemu

- Planowanie optymalnych tras podróży.
- Analiza i zarządzanie infrastrukturą drogową.
- Przewidywanie oraz zapobieganie korkom.
