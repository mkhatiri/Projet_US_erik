#!/usr/bin/env python3.5

import sys
from collections import defaultdict

def write_head(title="", with_title=False):
    """
    write the firste line of the latex documents
    """
    print("\documentclass{article} ")

    if with_title:
        print("\\title{"+title+"}")

    print("\\"+"begin{document}")
    if with_title:
        print("\\maketitle")


def write_end_file():
    print("\end{document}")




def write_table_per_graphe(graph_name, keys, values):
    """
    write table of graph
    """
    nb_keys = len(keys)

    print("\\textbf{"+graph_name+"}\\\\")

    print("\\begin{tabular}{|", end='')
    print(''.join( ['c|' for i in range(nb_keys)]), "}  ")

    print("\\hline")


    for i in range(nb_keys-1):
        print(keys[i], " & ", end='')

    print(keys[nb_keys-1], "\\\\")
    print("\\hline")


    for value in values:
        existing_keys = list(value.keys())
        for i in range(nb_keys-1):
            if keys[i] in existing_keys: 
                print(value[keys[i]], " & ", end='')
            else:
                print(" - ", "&", end='')
        if keys[nb_keys-1] in existing_keys: 
            print(value[keys[len(keys)-1]], "\\\\")
        else:
            print(" - ", "\\\\")

        print("\\hline")

    print("\\end{tabular}")



def run():

    keys = ['A', 'Age', 'med', 'walou', 'hmeeed']
    value1 = {'A': 'Zara', 'Age': 17};
    value2 = {'A': 'Med', 'Age': 27, 'hmed':'wah'};
    value3 = {'A': 'Khatiria', 'Age': 72, 'walou': 111};
    values = [value1, value2, value3];

    write_head(title="rapport", with_title=True)
    write_table_per_graphe("grapheX", keys, values)
    write_end_file()






if __name__ == "__main__":
    run()
