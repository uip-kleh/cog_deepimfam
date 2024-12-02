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
    setconfig = SetConfig(config_path)

    tools = Tools()
    entries = tools.load_aaindex1_entry("../common/aaindex1_entry.txt")

    for entry in entries:
        print("+++ CALCURATING {} PGM FILE +++".format(entry))
        image_generator = ImageGenerator(config_path, entry)
        image_generator.generate_images()
        image_generator.convert_pgm()