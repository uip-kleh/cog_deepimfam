import os
import sys
sys.path.append(os.pardir)
import json
import numpy as np
import pandas as pd
import tqdm 
import cv2
from common.tools import SetConfig, Tools

# +++ グラフ表示画像生成 +++
class ImageGenerator(SetConfig):
    def __init__(self, fname: str, entry: str) -> None:
        super().__init__(fname)
        if entry is None: return 
        self.entry = entry
        self.coordinates_path = self.join_path([self.results_path, entry, "coordinates"])
        self.images_path = self.join_path([self.results_path, entry, "images"])
        
    # +++ AAIndexの指標を読み込む +++
    def load_aaindex1(self):
        with open(self.aaindex1_path, "r") as f:
            self.aaindex1: dict = json.load(f)

    # +++ 標準化する +++
    def standarize(self, values: np.array) -> np.array:
        return (values - np.mean(values)) / np.std(values) 

    # +++ 標準化したベクトルを返す +++
    def generate_std_vectors(self) -> dict:
        self.load_aaindex1()
        keys = self.aaindex1[self.entry].keys()
        values1 = np.array(list(self.aaindex1[self.entry].values()))
        values2 = np.array([0.1 for i in range(len(self.aaindex1[self.entry].values()))])
        std_values1 = self.standarize(values1)

        vectors = {}
        for i, key in enumerate(keys):
            vectors[key] = [std_values1[i], values2[i]]

        return vectors

    # +++ アミノ酸配列を読み込む +++
    def load_sequences(self) -> list:
        sequences = []
        with open(self.amino_data_path, "r") as f:
            for l in f.read().split("\n"):
                if l == "": continue
                label, seq = l.split()
                if label == "0" or label == "1":
                    sequences.append(seq)
                
        return sequences

    # +++ ラベルを読み込む +++
    def load_labels(self) -> list:
        labels = []
        with open(self.amino_data_path, "r") as f:
            for l in f.read().split("\n"):
                if l == "": continue
                label, seq = l.split()
                labels.append(label)
                
        return labels

    # +++ 座標を計算する +++
    def calc_coordinates(self):
        vectors = self.generate_std_vectors()
        sequences = self.load_sequences()
        
        for i, seq in enumerate(tqdm.tqdm(sequences)):
            fname = os.path.join(self.coordinates_path, str(i) + ".dat")
            with open(fname, "w") as f:
                x, y = 0, 0
                print("{}, {}".format(x, y), file=f)
                for aa in seq:
                    if not aa in vectors:
                        continue
                    x += vectors[aa][0]
                    y += vectors[aa][1]
                    print("{}, {}".format(x, y), file=f)

    def convert_pgm(self):
        for i in tqdm.tqdm(range(self.NUM_DATA)):
            pgm_fname = os.path.join(self.images_path, str(i) + ".pgm")
            png_fname = os.path.join(self.images_path, str(i) + ".png")

            cv2.imwrite(png_fname, cv2.imread(pgm_fname))

    def make_image_info(self):
        labels = []
        path = []
        labels = [int(label) for label in self.load_labels()]
        for i in tqdm.tqdm(range(len(labels))):
            fname = os.path.join(self.images_path, str(i) + ".png")
            # print(label, fname)
            path.append(fname)

        print(len(labels), len(path))
        fname = os.path.join(self.images_path, "images_info.csv")
        pd.DataFrame({
            "labels": labels,
            "image_path": path,
        }).to_csv(fname)
            

if __name__ == "__main__":
    config_path = "config.yaml"
    setconfig = SetConfig(config_path)

    tools = Tools()
    entries = tools.load_aaindex1_entry("../common/aaindex1_entry.txt")
    
    for entry in entries:
        print("+++ CALCURATING {} COORDINATE +++".format(entry))
        image_generator = ImageGenerator(config_path, entry)
        image_generator.calc_coordinates()
        # image_generator.make_image_info()