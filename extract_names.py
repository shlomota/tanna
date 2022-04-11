import pandas as pd
import os
import re
from collections import Counter

BASE_PATH = r"C:\Users\soki\Documents\TAU\DL\proj"
df = pd.read_json(os.path.join(BASE_PATH, r"mishna\dataset.json"))

start_list = ["הלל הזקן", "שמאי הזקן", "בית הלל", "בית שמאי", "חכמים"]
res = ["(רב[ין] [א-ת]*) (?!בן)",
       "(רב[ין] [א-ת]* בן [א-ת]*) (?!ה)",
       "([א-ת]* בן [א-ת]*)(?: אומר)",
       "(?:(?:דברי|אמר)) [א-ת]* בן [א-ת]*",
       "(?:דברי) (?!תורה)([א-ת]* [א-ת]*) (?!בן)",
       "([א-ת]* (בן )?[א-ת])(?: אומר)",
       "רבי[ין] [א-ת]* בן [א-ת]* ה[א-ת]?",
       "רב[ין]? [א-ת]* ה[א-ת]*"]

if not os.path.exists("names.txt"):
    name_list = []
    for i, row in df.iterrows():
        for regex in res:
            matches = re.findall(regex, row.text)
            name_list += matches
            if len(matches) > 0 and "שמעו" == matches[0][-4:]:
                a=5

    name_list += start_list
    result = Counter(name_list)
    print(result)

    with open("names.txt", "w") as f:
        for name in result:
            f.write(name + "\n")

with open("names.txt", "r") as f:
    names = f.readlines()
names = [name.strip() for name in names]
names = list(set(names))
names = [name.replace(" ", "_") for name in names]
