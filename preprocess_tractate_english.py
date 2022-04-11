import re
import glob
import os
import pandas as pd
import numpy as np
from pathlib import Path

seder2int = {"zeraim": 1,
             "moed": 2,
             "nashim": 3,
             "nezikin": 4,
             "kodashim": 5,
             "tahorot": 6}

BASE_PATH = r"C:\Users\soki\Documents\TAU\Thesis\Mishnah"
LINE_MIN_LEN = 20
paths = glob.glob(os.path.join(BASE_PATH, r"*\*\English\merged.txt"))

all_lines = []
all_seders = []
all_tractates = []
all_chapters = []
all_indices = []

for path in paths:
    seder_name = Path(path).parents[2].name.lower().split()[-1]
    seder = seder2int[seder_name]
    # seder = int(os.path.basename(os.path.dirname(path)))
    base = os.path.basename(path)
    tractate = " ".join(Path(path).parents[1].name.lower().split()[1:])
    # tractate = base[base.find("_")+1:].split(".")[0]
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()#[:10000]

    if "Chapter" in data:
        data = data[data.find("Chapter"):]

    chapter = 1
    while "Chapter" in data:
        cur_data = data
        end_idx = data.find("Chapter", data.find("Chapter") + 1)
        if end_idx != -1:
            cur_data = data[:end_idx]
        data = data[end_idx:]

        # result = re.sub(":", ".", cur_data)
        result = re.sub("[\[\]]", "", cur_data)
        result = re.sub("\(.*?\) ", "", result)
        # result = re.sub("[^א-ת\.\,\s]", "", result) #no nikkud
        # result = re.sub("[^ְֱֲֳִֵֶַָֹֺֻא-ת\.\,\s]", "", result) #with nikkud
        # result = re.sub("\.", "\n", result)
        # result = re.sub("\n\n", "\n", result)
        # result = result.replace("\n\n", "\n")
        lines = result.split("\n")
        lines = [line for line in lines if len(line) > LINE_MIN_LEN]

        all_lines += lines
        all_seders += [seder] * len(lines)
        all_tractates += [tractate] * len(lines)
        all_chapters += [chapter] * len(lines)
        all_indices += list(range(1, len(lines)+1))


        chapter += 1

df = pd.DataFrame(data=np.array([all_lines, all_seders, all_tractates, all_chapters, all_indices]).T, columns=["text", "seder", "tractate", "chapter", "index"])
# df.to_json(r"C:\Users\soki\Documents\TAU\DL\proj\mishna\dataset.json")
df.to_json(os.path.join(BASE_PATH, r"dataset_en.json"))
a=5