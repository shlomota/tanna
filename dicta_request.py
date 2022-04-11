import requests
import json

headers = {'Content-Type': 'text/plain;charset=utf-8'}
params = {
    "task" : "nakdan",
    "genre" :"modern",
    "data" : "מאימתי קורין את שמע בערבית? משעה שהכהנים נכנסים לאכל בתרומתן",
    "addmorph" : True,
    "matchpartial" : True,
    "keepmetagim" : False,
    "keepqq" :False,
}
# r = requests.post("https://nakdan-3-2.loadbalancer.dicta.org.il/addnikud",headers=headers,json=params)
# r.encoding= "UTF-8"
# print(r.text)


def get_stemmed(text):
    params = {
        "task" : "nakdan",
        "genre" :"rabbinic",
        # "genre" :"modern",
        "data" : text,
        "addmorph" : True,
        "matchpartial" : True,
        "keepmetagim" : False,
        "keepqq" :False,
    }
    r = requests.post("https://nakdan-3-2.loadbalancer.dicta.org.il/addnikud",headers=headers,json=params)
    r.encoding= "UTF-8"
    res = json.loads(r.text)
    words = [word["options"][0]["lex"] if len(word["options"]) > 0 else word["word"] for word in res]
    result = "".join(words)
    return result

def get_diacriticized(text):
    params = {
        "task" : "nakdan",
        "genre" :"rabbinic",
        # "genre" :"modern",
        "data" : text,
        "addmorph" : True,
        "matchpartial" : True,
        "keepmetagim" : False,
        "keepqq" :False,
    }
    r = requests.post("https://nakdan-3-2.loadbalancer.dicta.org.il/addnikud",headers=headers,json=params)
    r.encoding= "UTF-8"
    res = json.loads(r.text)
    words = [word["options"][0]["w"] if len(word["options"]) > 0 else word["word"] for word in res]
    result = "".join(words)
    return result

def get_morph(text):
    params = {
        "task" : "nakdan",
        "genre" :"rabbinic",
        # "genre" :"modern",
        "data" : text,
        "addmorph" : True,
        "matchpartial" : True,
        "keepmetagim" : False,
        "keepqq" :False,
    }
    r = requests.post("https://nakdan-3-2.loadbalancer.dicta.org.il/addnikud",headers=headers,json=params)
    r.encoding= "UTF-8"
    res = json.loads(r.text)
    res = filter(lambda x: len(x["options"]) > 0, res)
    words = [word["options"][0]["morph"] for word in res]
    result = " ".join(words)
    return result


def get_text_morph(text):
    params = {
        "task" : "nakdan",
        "genre" :"rabbinic",
        # "genre" :"modern",
        "data" : text,
        "addmorph" : True,
        "matchpartial" : True,
        "keepmetagim" : False,
        "keepqq" :False,
    }
    r = requests.post("https://nakdan-3-2.loadbalancer.dicta.org.il/addnikud",headers=headers,json=params)
    r.encoding= "UTF-8"
    res = json.loads(r.text)
    res = list(filter(lambda x: x["word"] != " ", res))
    words = [word["word"].strip() for word in res]
    diacs = [word["options"][0]["w"] if len(word["options"]) > 0 else "" for word in res]
    lexes = [word["options"][0]["lex"] if len(word["options"]) > 0 else "" for word in res]
    morphs = [word["options"][0]["morph"] if len(word["options"]) > 0 else "" for word in res]
    return words, morphs, diacs, lexes
    # return result

if __name__ == "__main__":
    # get_stemmed("מאימתי קורין את שמע בערבית? משעה שהכהנים נכנסים לאכל בתרומתן")
    # print(get_diacriticized("מאימתי קורין את שמע בערבית? משעה שהכהנים נכנסים לאכל בתרומתן"))
    print(get_diacriticized("תנא היכא קאי דקתני מאימתי ותו מאי שנא דתני בערבית ברישא לתני דשחרית ברישא תנא אקרא קאי דכתיב (דברים ו, ז) בשכבך ובקומך והכי קתני זמן קריאת שמע דשכיבה אימת משעה שהכהנים נכנסין לאכול בתרומתן ואי בעית אימא"))
    print(get_morph("תנא היכא קאי דקתני מאימתי ותו מאי שנא דתני בערבית ברישא לתני דשחרית ברישא תנא אקרא קאי דכתיב (דברים ו, ז) בשכבך ובקומך והכי קתני זמן קריאת שמע דשכיבה אימת משעה שהכהנים נכנסין לאכול בתרומתן ואי בעית אימא"))
    # print(get_morph("מאימתי קורין את שמע בערבית? משעה שהכהנים נכנסים לאכל בתרומתן"))