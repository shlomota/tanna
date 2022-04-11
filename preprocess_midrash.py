import re
import glob
import os
import pandas as pd
import numpy as np
from preprocess_mishna_tosefta import clean_data

BASE_PATH = r"C:\Users\soki\Documents\TAU\DL\proj"
LINE_MIN_LEN = 20
paths = glob.glob(os.path.join(BASE_PATH, r"midrash\*.txt"))

all_lines = []
all_sources = []
all_chapters = []
all_indices = []
name_to_idx = {"shimon":0, "yishmael":1}

for path in paths:
    base = os.path.basename(path)
    source = name_to_idx[base[base.find("_")+1:].split(".")[0]]
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()#[:10000]

    chapter = 1
    while "Chapter" in data:
        cur_data = data
        end_idx = data.find("Chapter", data.find("Chapter") + 1)
        if end_idx != -1:
            cur_data = data[:end_idx]
        data = data[end_idx:]

        result = re.sub(":", ".", cur_data)
        result = re.sub("\(.*?\) ", "", result)
        result = re.sub("\[.*?\] ", "", result)
        result = clean_data(result)
        # result = re.sub("\.", "\n", result)
        # result = re.sub("\n\n", "\n", result)
        # result = result.replace("\n\n", "\n")
        lines = result.split("\n")
        lines = [line for line in lines if len(line) > LINE_MIN_LEN]

        all_lines += lines
        all_sources += [source] * len(lines)
        all_chapters += [chapter] * len(lines)
        all_indices += list(range(1, len(lines)+1))


        chapter += 1

df = pd.DataFrame(data=np.array([all_lines, all_sources, all_chapters, all_indices]).T, columns=["text", "source", "chapter", "index"])
df.to_json(os.path.join(BASE_PATH, r"midrash\dataset.json"))
a=5