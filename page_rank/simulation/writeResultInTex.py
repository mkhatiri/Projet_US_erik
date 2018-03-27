#!/usr/bin/env python3.5

import sys
from collections import defaultdict

def write_head(title="", with_title=False):
    """
    write the firste line of the latex documents
    """
    print("\documentclass[9pt]{article} ")

    add_package()
    if with_title:
        print("\\title{"+title+"}")

    print("\\"+"begin{document}")
    if with_title:
        print("\\maketitle")


def write_end_file():
    print("\end{document}")



def add_package():
    """
    print all packeges used in latex
    """
    print("\\pagestyle{empty}")
    print("\\usepackage[left=3cm,right=2cm,top=0.5cm,bottom=0.5cm]{geometry}")
    print("\\usepackage[all]{background}")
    print("\\usepackage{lipsum}")


def add_table_title(table_name, host):
    """
        add table name (vertical)
    """
    print("\\SetBgContents{"+table_name[5:-11]+" on "+host[:-1]+"}")
    print("\\SetBgPosition{0.25cm,-5.0cm}")
    print("\\SetBgOpacity{1.0}")
    print("\\SetBgAngle{90.0}")
    print("\\SetBgScale{2.0}")

def write_table_per_graphe(keys, values):
    """
    write table of graph
    """
    nb_keys = len(keys)


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




def load_file(file_name, program_name=""):
    """
    load file and stor it on list of dict
    """
    all_value = list()
    value = dict()


    value["program"] =  program_name


    f=open(file_name, 'r')
    line = f.readline()

    if line:
        host = line

    line = f.readline()
    while line:
        split_line = line.split(" ")
        for i in range(0, len(split_line)-1, 2):
            value[str(split_line[i][:-1])] =  split_line[i+1]

        all_value.append(value)
        value = {}
        line = f.readline()

    return  host, all_value



def load_all_file(names):
    """
        load all files on one pdf file
    """
    for file_name in names:
        load_file("test/"+file_name+".o.data")

def run(files_names):

    keys = ['nVtx', 'nonzero',  'rowBlockSize', 'shortBlockSize', 'BlkSize', 'nThreadPerBlock',   'AvgTime']



    write_head(title="rapport", with_title=False)
    for i in range(1,len(files_names)-2):
        host, values = load_file(files_names[i])
        add_table_title(files_names[i], host)
        write_table_per_graphe(keys, values)
        print(" \n")
    write_end_file()



if __name__ == "__main__":
    run(sys.argv)
