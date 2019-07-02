__author__ = 'Horace'

import sys, os, setting
import pandas

def read(filename):
    working_path = setting.get_working_path()
    ext = filename.split('.')
    type = ext[len(ext)-1]
    if type == "csv":
        raw_data = pandas.read_csv(os.path.join(working_path, "data_sources", filename))
    elif type == "json":
        raw_data = pandas.read_json(os.path.join(working_path, "data_sources", filename))
    else:
        raise TypeError("wrong file format inputted")

    print raw_data
    return raw_data









