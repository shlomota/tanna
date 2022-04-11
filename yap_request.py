import json
import pandas as pd
import numpy as np
import requests
import io


host = "c-004"
port = 8000
yap_address = f"http://{host}:{port}/yap/heb/joint"
headers = {'content-type': 'application/json'}

def get_stemmed(text):
    data = '{{"text": "{}  "}}'.format(text).encode('utf-8')  # input string ends with two space characters
    response = requests.get(url=yap_address, data=data, headers=headers)
    json_response = response.json()
    result = json_response["md_lattice"]
    df = pd.read_csv(io.StringIO(result), sep="\t", names=["a", "b", "c", "stemmed", "e", "f", "g", "h"])
    return " ".join(df.stemmed)
