import re
import glob
import os
import pandas as pd
import numpy as np

LINE_MIN_LEN = 20
mishna_paths = glob.glob(r"C:\Users\soki\Documents\TAU\DL\proj\mishna\unprocessed\*\*")
tosefta_paths = glob.glob(r"C:\Users\soki\Documents\TAU\DL\proj\mishna\tosefta\*")

acronyms = {
    "אע\"פ": "אף על פי",
    "שנא'": "שנאמר",
    "ב\"ש": "בית שמאי",
    "ב\"ה": "בית הלל",
    "א\"ר": "אמר רבי",
    "ר'": "רבי",
    "עקיבה": "עקיבא",
    "רשב\"ג": "רבן שמעון בן גמליאל",
    "חכ\"א": "חכמים אומרים",
    "ר\"ש": "רבי שמעון",
    "ר\"א": "רבי אליעזר",
    "ב\"ר": "בן רבי",
    "אח\"כ": "אחר כך",
    "בד\"א": "במה דברים אמורים",
    "ב\"ד": "בית דין",
    "אפי'": "אפילו",
    "א'": "אחד",
    "ע\"ג": "על גבי",
    "ר\"ע": "רבי עקיבא",
    "יו\"ט": "יום טוב",
    "ת\"ל": "תלמוד לומר",
    "ע\"מ": "על מנת",
    "אא\"כ": "אלא אם כן",
    "רה\"ר": "רשות הרבים",
    "רה\"י": "רשות היחיד",
    "רשב\"א": "רבי שמעון בן אלעזר",
    "ע\"י": "על ידי",
    "בעה\"ב": "בעל הבית",
    "ע\"פ": "על פי",
    "א\"צ": "אין צריך",
    "ב\"א": "בן אלעזר",
    "ר\"ג": "רבן גמליאל",
    "ריב\"ז": "רבן יוחנן בן זכאי",
    "א\"כ": "אם כן",
    "ר\"י": "רבי יהודה",
    "אעפ\"כ": "אף על פי כן",
    "ע\"ה": "עם הארץ",
    "רש\"א": "רבי שמעון אומר",
    "ק\"ו": "קל וחומר",
    "יוה\"כ": "יום הכיפורים",
    "י\"ט": "יום טוב",
    "ע\"ש": "ערב שבת",
    "נ\"ש": "נזק שלם",
    "ח\"נ": "חצי נזק",
    "ה\"ז": "הרי זה",
    "ד\"א": "דבר אחר",
    "יהושוע": "יהושע",
    "ר\"מ": "רבי מאיר",
    "חביר": "חבר",
    "אפלו": "אפילו",
    "חיב": "חייב",
    "מניין": "מנין",
    "שנ'": "שנאמר"

}

def clean_data(data):
    data = re.sub("[^א-ת\.\,\'\":\s]", "", data)
    data = re.sub(":", ".", data)
    for k in acronyms:
        data = data.replace(k, acronyms[k])
    return data

def main():
    all_lines = []
    all_labels = []

    for path in mishna_paths:
        label = 1
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()

        result = clean_data(data)

        # split sentences
        result = result.replace("\n", "")
        lines = result.split(".")
        lines = [line + "." for line in lines if len(line) > LINE_MIN_LEN]

        all_lines += lines
        all_labels += [label] * len(lines)


    for path in tosefta_paths:
        label = 2
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()#[:10000]

        result = clean_data(data)

        result = result.replace("\n", "")
        lines = result.split(".")
        lines = [line for line in lines if len(line) > LINE_MIN_LEN]

        all_lines += lines
        all_labels += [label] * len(lines)

    df = pd.DataFrame(data=np.array([all_lines, all_labels]).T, columns=["text", "label"])
    df.to_json(r"C:\Users\soki\Documents\TAU\DL\proj\mishna\tosefta_dataset.json")


if __name__ == "__main__":
    main()