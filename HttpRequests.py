from typing import List
import requests

class State:
    def __init__(self):
        self.actions = []
        

class RoseAI_Interface:
    def __init__(self):
        pass

    def GetDataFromDatabase(self):
        try:
            response = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
            with open("helper_functions.py", "wb") as f:
                f.write(response.content)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    connect = RoseAI_Interface()
    connect.GetDataFromDatabase()
        
        
