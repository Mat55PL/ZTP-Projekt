import pandas as pd

# DataLoader: klasa odpowiedzialna za wczytywanie danych z pliku CSV
class DataLoader: 
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self):
        data = pd.read_csv(self.file_path)
        return data