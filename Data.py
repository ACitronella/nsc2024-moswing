import os
import json
from datetime import date, datetime

class JsonDataStore:
    def __init__(self, DB_PATH):
        self.DB_PATH = DB_PATH
        if not os.path.exists(DB_PATH) and not os.path.isfile(DB_PATH):
            raise FileNotFoundError(f"{DB_PATH} is not file or not exist.") # maybe create a file here with "{}" as a content
        self.obj = load_data_instance(DB_PATH)

def load_data_instance(DB_PATH:str) -> dict:
    with open(DB_PATH, mode="r") as f:
        j = json.load(f)    
    return j

def store_data_instance(db:JsonDataStore):
    with open(db.DB_PATH, mode="w") as f:
        json.dump(db.obj, f)

def add_location_to_data_instance(db:JsonDataStore, file_name:str, location:dict, date_:date):
    d = {
            "location": location,
            "timestamp": datetime(date_.year, date_.month, date_.day).timestamp()
    }
    db.obj[file_name] = d
    store_data_instance(db)
    print("done saving")
