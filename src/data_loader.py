import pandas as pd
import kagglehub
class DataLoader:
    def __init__(self):
        pathKaggleHub = kagglehub.dataset_download("bwandowando/tomtom-traffic-data-55-countries-387-cities")

        print("[INFO] Ścieżka do dataset:", pathKaggleHub)
        finalPath = pathKaggleHub + '\ForExportNewApi.csv\ForExportNewApi.csv'
        self.path = finalPath

    def load_data(self):
        data = pd.read_csv(self.path)
        return data
