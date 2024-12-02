import os
import requests
import json

class AAindex:
    url = "https://www.genome.jp/ftp/db/community/aaindex"

    def __init__(self) -> None:
        pass

    def get_aaindex1(self):
        aaindex1_url = os.path.join(self.url, "aaindex1")
        query = requests.get(aaindex1_url).content.decode('utf-8').split("\n")

        line1_columns = "ARNDCQEGHI"
        line2_columns = "LKMFPSTWYV"

        aaindex1 = {}

        record = None
        for i in range(len(query)):
            if len(query[i]) == 0:
                continue
            if query[i][0] == "H":
                record = query[i].split()[1]
            if query[i][0] == "I":
                if "NA" in query[i+1] or "NA" in query[i+2]:
                    continue

                splited_line1 = list(map(float, query[i+1].split()))
                splited_line2 = list(map(float, query[i+2].split()))

                index = {}
                for i in range(10):
                    index[line1_columns[i]] = splited_line1[i]
                    index[line2_columns[i]] = splited_line2[i]

                aaindex1[record] = index


        with open("aaindex1.json", "w") as f:
            json.dump(aaindex1, f, indent=2)

if __name__ == "__main__":
    aaindex = AAindex()
    aaindex.get_aaindex1()