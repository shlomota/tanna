import re
import glob
import os
import pandas as pd
import numpy as np
# from yap_request import get_stemmed
from dicta_request import get_stemmed
from tqdm import tqdm

# BASE_PATH = r"/a/home/cc/students/cs/shlomotannor/tanna/"
BASE_PATH = r"C:\Users\soki\Documents\TAU\DL\proj"
LINE_MIN_LEN = 20
# paths = glob.glob(os.path.join(BASE_PATH, r"mishna/unprocessed/*/*"))
paths = glob.glob(os.path.join(BASE_PATH, r"mishna\unprocessed\*\*"))

all_lines = []
all_seders = []
all_tractates = []
all_chapters = []
all_indices = []

for path in tqdm(paths):
    seder = int(os.path.basename(os.path.dirname(path)))
    base = os.path.basename(path)
    tractate = base[base.find("_")+1:].split(".")[0]
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

        result = re.sub(":", ".", cur_data)
        # result = cur_data
        result = re.sub("\(.*?\) ", "", result)
        # result = re.sub("[\[\]]", "", result)
        result = re.sub("[^א-ת\.\,\s]", "", result) #no nikkud
        # result = re.sub("[^ְֱֲֳִֵֶַָֹֺֻּא-ת\.\,\s]", "", result) #with nikkud
        # result = re.sub("\.", "\n", result)
        # result = re.sub("\n\n", "\n", result)
        # result = result.replace("\n\n", "\n")
        lines = result.split("\n")
        lines = [line for line in lines if len(line) > LINE_MIN_LEN]
        lines = [line.replace(",", " ,").replace(".", " .") for line in lines]
        lines = [get_stemmed(line) for line in lines]

        all_lines += lines
        all_seders += [seder] * len(lines)
        all_tractates += [tractate] * len(lines)
        all_chapters += [chapter] * len(lines)
        all_indices += list(range(1, len(lines)+1))
        # if chapter % 10 == 0:
        print(chapter, flush=True)
        chapter += 1


df = pd.DataFrame(data=np.array([all_lines, all_seders, all_tractates, all_chapters, all_indices]).T, columns=["text", "seder", "tractate", "chapter", "index"])
df.to_json(r"dataset_stemmed_dicta.json")
# df.to_json(r"C:\Users\soki\Documents\TAU\DL\proj\mishna\dataset_stemmed.json")
# df.to_json(os.path.join(BASE_PATH, r"mishna\dataset_stemmed.json"))
# a=5