import re
import glob
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from preprocess_style import acronyms, clean
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

BASE_PATH = r"/a/home/cc/students/cs/shlomotannor/tanna/style/filtered"
# BASE_PATH = r"C:\Users\soki\Documents\TAU\DL\proj\style\filtered"
LINE_MIN_LEN = 20
# paths = glob.glob(os.path.join(BASE_PATH, r"*\*\Hebrew\merged.txt"))
paths = glob.glob(os.path.join(BASE_PATH, r"*/*/Hebrew/merged.txt"))
# paths += glob.glob(os.path.join(BASE_PATH, r"*\*\*\Hebrew\merged.txt"))
paths += glob.glob(os.path.join(BASE_PATH, r"*/*/*/Hebrew/merged.txt"))
# paths += glob.glob(os.path.join(BASE_PATH, r"*\*\*\*\Hebrew\merged.txt"))
paths += glob.glob(os.path.join(BASE_PATH, r"*/*/*/*/Hebrew/merged.txt"))
paths = sorted(paths)
print(len(paths))

do_morph = True
do_diac = False

def filter_line(line):
    if len(line) < LINE_MIN_LEN:
        return False
    if len(line.split()) < 5:
        return False
    if not re.search("[א-תםןץףך]", line):
        return False
    if re.search("[a-zA-Z]", line):
        return False
    return True



CHUNK_SIZE = 50

all_chunks = []
all_labels = []
all_books = []
all_chunk_ids = []
from dicta_request import get_diacriticized, get_morph

for path in tqdm(paths):
    # label = int(path[len(BASE_PATH):].split("\\")[1])
    label = int(path[len(BASE_PATH):].split("/")[1])
    # book = path[len(BASE_PATH):].split("\\")[2]
    book = path[len(BASE_PATH):].split("/")[2]
    base = os.path.basename(path)
    tractate = base[base.find("_")+1:].split(".")[0]

    with open(path, "r", encoding="utf-8") as f:
        data = f.read()#[:10000]


    lines = data.split("\n")
    lines = list(filter(filter_line, lines))
    data = "\n".join(lines)
    data = clean(data)

    data = re.sub("שנאמר [א-תםןףךץ]+ [א-תםןףךץ]+,", "שנאמר", data)

    data = data.split()
    chunks = [' '.join(data[i:i + CHUNK_SIZE]) for i in range(0, len(data), CHUNK_SIZE)]

    chunks_df = pd.Series(name="text", data=chunks)
    if do_morph:
        chunks_df = chunks_df.parallel_apply(get_morph)
        # chunks = [get_morph(chunk) for chunk in tqdm(chunks)]
    elif do_diac:
        chunks_df = chunks_df.parallel_apply(get_diacriticized)
        # chunks = [get_diacriticized(chunk) for chunk in tqdm(chunks)]
    chunks = chunks_df.values.tolist()


    all_chunks += chunks
    all_labels += [label] * len(chunks)
    all_books += [book] * len(chunks)
    all_chunk_ids += range(len(chunks))

df = pd.DataFrame(data=np.array([all_chunks, all_labels, all_books, all_chunk_ids]).T, columns=["text", "label", "book", "chunk_id"])
if do_morph:
    # df.to_json(r"C:\Users\soki\Documents\TAU\DL\proj\style\dataset_morph.json")
    df.to_json(r"/a/home/cc/students/cs/shlomotannor/tanna/style/dataset_morph.json")
elif do_diac:
    # df.to_json(r"C:\Users\soki\Documents\TAU\DL\proj\style\dataset_diac.json")
    df.to_json(r"/a/home/cc/students/cs/shlomotannor/tanna/style/dataset_diac.json")
# df.to_json(os.path.join(BASE_PATH, r"mishna\dataset.json"))
