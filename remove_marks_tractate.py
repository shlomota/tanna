import re
import glob
import os
import pandas as pd
import numpy as np

LINE_MIN_LEN = 20
paths = glob.glob(r"C:\Users\soki\Documents\TAU\DL\proj\mishna\processed\*\*")

all_lines = []
all_labels = []

for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()#[:10000]

    result = re.sub(":", ".", data)
    result = re.sub("[^א-ת\.\,\s]", "", result)
    # result = re.sub("\.", "\n", result)
    # result = re.sub("\n\n", "\n", result)
    # result = result.replace("\n\n", "\n")
    lines = result.split("\n")
    lines = [line for line in lines if len(line) > LINE_MIN_LEN]
    data = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)#[:10000]
