import os
import sys
import yaml
import pandas as pd
import matplotlib  # <--追記
matplotlib.use('Agg')  # <--追記
import json

class SetConfig:
    def __init__(self, fname) -> None:
        # +++ config.yamlを読み込む +++
        with open(fname, "r") as f:
            args = yaml.safe_load(f)
            self.image_method = args["image_method"]
            self.amino_data_path = args["amino_data_path"]
            self.dataset = args["dataset"]
            self.aaindex1_entry_path = args["aaindex1_entry_path"]
            self.aaindex1_path = args["aaindex1_path"]
            self.results_path = args["results_path"]
            self.model_method = args["model_method"]
            self.IMAGE_SIZE = args["IMAGE_SIZE"]
            self.NUM_DATA = args["NUM_DATA"]
            self.NUM_CLASS = args["NUM_CLASS"]
            self.BATCH_SIZE = args["BATCH_SIZE"]
             
        self.amino_data_path = self.join_path([self.amino_data_path, self.dataset, "data.txt"])
        self.results_path = self.join_path([self.results_path, self.image_method])

    # +++ パス（文字列）をつなげる +++
    def join_path(self, path_list: list) -> str:
        path = path_list[0]
        for s in path_list[1:]:
            path = os.path.join(path, s)
            self.make_dir(path)
        return path
    
    # +++ ディレクトリを作成する +++
    def make_dir(self, path: str) -> str:
        if not os.path.exists(path): os.mkdir(path)
        return path

    # +++ オブジェクトをJSON形式で保存 +++ 
    def save_obj(self, obj, fname):
        with open(fname, "w") as f:
            json.dump(obj, f, indent=2)

    def save_dict_as_dataframe(self, obj: dict, fname):
        pd.DataFrame(obj).to_csv(fname, index_label=False)

    def load_csv_as_dict(self, fname):
        return pd.read_csv(fname).to_dict(orient="list")
    
class Tools:
    def __init__(self) -> None:
        pass
    
    def load_aaindex1_entry(self, fname):
        entries = []
        with open(fname, "r") as f:
            for entry in f.read().split("\n"):
                entries.append(entry)
                
        return entries

if __name__ == "__main__":
    pass