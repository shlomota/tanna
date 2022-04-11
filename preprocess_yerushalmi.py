import re
import os
import glob
from bs4 import BeautifulSoup
from pprint import pprint

mode = "Bavli"


BASE_PATH = r"C:\Users\soki\Documents\TAU\DL\proj\style\r002\r"
if mode == "Bavli":
    BASE_PATH = r"C:\Users\soki\Documents\TAU\DL\proj\style\l002\b\l"
paths = glob.glob(os.path.join(BASE_PATH, "*.htm"))

print(len(paths))
pprint(sorted(paths))

seder_list = ["Zeraim", "Moed", "Nashim", "Nezikin", "Kodashim", "Tahorot"]
for path in paths:
    filename = path.split("\\")[-1][:-4]
    seder = seder_list[int(filename[1]) - 1]
    # os.mkdir(os.path.join(BASE_PATH, f"Seder {seder}"))

    with open(path, "r") as f:
        data = f.read()

    data = re.sub("<p>[^p]*?<b>[^b]*משנה.*?</p>", "", data, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
    # a = re.search("<p>[^p]*?<b>[^b]*משנה.*?</p>", data, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
    # data2 = re.match("<p>.*?<b>[^b]*משנה.*</p>", data, re.MULTILINE | re.DOTALL)
    soup = BeautifulSoup(data)
    tags = ["b"]
    for tag in tags:
        for s in soup.select(tag):
            s.extract()
    text = soup.get_text()
    tractate = re.search("מסכת (.*) פרק", text).group(1)

    dir = os.path.join(os.path.join(os.path.join(BASE_PATH, f"Seder {seder}"), f"מסכת {tractate}"), "Hebrew")
    os.makedirs(dir)

    with open(os.path.join(dir, "merged.txt"), "w", encoding="utf8") as f:
        f.write(text)

    a = 5