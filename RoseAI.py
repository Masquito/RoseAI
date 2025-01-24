from typing import List
import requests


class RoseAI_Interface:
    def __init__(self):
        pass

    def GetDataFromDatabase(self, url: str):
        try:
            cert_path = "cert.cer"
            text = {'somekey': 'somevalue'}
            response = requests.post(url,json=text, verify=cert_path)
            print(response.text)
            print(response.status_code)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    url = "https://localhost:7202/api/Main/grid"
    rose = RoseAI_Interface()
    rose.GetDataFromDatabase(url)