import csv
import re

import pandas as pd


def create_file(path, new_path, num=2):
    a = []
    for line in open(path):
        a.append(re.split(r"\s|\t", line[:-1]))

    if num == 2:
        with open(new_path, "w") as file:
            for line in a:
                file.write("{}\t{}\n".format(line[0], " ".join(line[1:])))
    if num == 3:
        with open(new_path, "w") as file:
            for line in a:
                file.write(
                    "{}\t{}\t{}\n".format(line[0], " ".join(line[1:-1]), line[-1])
                )


def read_file(path, names=["id", "text", "label"]):
    data = pd.read_table(path, header=0, names=names, sep="\t", quoting=csv.QUOTE_NONE, encoding="utf-8").applymap(str)

    return data

