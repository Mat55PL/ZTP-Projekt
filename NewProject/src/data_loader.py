import pandas as pd

class DataLoader:
    def __init__(self, path):
        self.path = path

    def load_data(self):
        data = pd.read_csv(self.path)
        return data
