import re
import glob
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

BASE_PATH = r"C:\Users\soki\Documents\TAU\DL\proj\style\filtered"
LINE_MIN_LEN = 20

df = pd.read_json(r"C:\Users\soki\Documents\TAU\DL\proj\style\dataset.json")
df_morph = pd.read_json(r"C:\Users\soki\Documents\TAU\DL\proj\style\dataset_morph.json")
a = 5