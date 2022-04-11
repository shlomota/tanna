import pandas as pd
import os
BASE_PATH = r"C:\Users\soki\Documents\TAU\DL\proj"

df = pd.read_json(os.path.join(BASE_PATH, r"mishna\dataset.json"))