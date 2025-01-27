from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        # n_estimators: liczba drzew im więcej tym lepiej, ale zwiększa to czas trenowania
        # random_state: ziarno losowości w modelu - pozwala na odtworzenie wyników
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def train(self, X_train, y_train):
        # X_train - zbiór treningowy z danymi wejściowymi 
        # y_train - zbiór treningowy z danymi wyjściowymi (target)
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        # X_test - zbiór testowy z danymi wejściowymi
        # y_test - zbiór testowy z danymi wyjściowymi (target)
        # zwraca mse - błąd średniokwadratowy oraz r2 - współczynnik determinacji
        # błąd średniokwadratowy to średnia kwadratów różnic pomiędzy wartościami przewidywanymi przez model a wartościami rzeczywistymi
        # współczynnik determinacji to miara określająca jakość dopasowania modelu do danych
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2
    
    def predict(self, X):
        # X - dane wejściowe dla których chcemy dokonać predykcji
        # zwraca predykcje modelu
        return self.model.predict(X)