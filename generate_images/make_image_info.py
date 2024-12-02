import os
import sys
sys.path.append(os.pardir)
import json
import numpy as np
import tqdm 
from common.tools import SetConfig, Tools
from calcurate_coordinates import ImageGenerator

if __name__ == "__main__":
    config_path = "config.yaml"
    tools = Tools()
    entries = tools.load_aaindex1_entry("../common/aaindex1_entry.txt")
    

    print("+++ MAKING IMAGES INFO +++")
    for entry in entries:
        image_generator = ImageGenerator(config_path, entry)
        image_generator.make_image_info()