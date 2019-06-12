# -*- coding: utf-8 -*-

import sys

python_version = list(sys.version_info)
if python_version[0] != 3 or (python_version[0] == 3 and python_version[1] < 7):
    print("Your Python version does not meet requirements. (Python >3.7)")
    exit(1)
else:
    print("Your Python version satisfies the requirements.")
try:
    import keras
    print("Importing keras successful")
except:
    print("Importing keras failed!")
try:
    import tensorflow
    print("Importing tensorflow successful")
except:
    print("Importing tensorflow failed!")
try:
    import numpy
    print("Importing numpy successful")
except:
    print("Importing numpy failed!")
try:
    import pandas
    print("Importing pandas successful")
except:
    print("Importing pandas failed!")
try:
    import matplotlib.pyplot as plt
    print("Importing matplotlib successful")
except:
    print("Importing matplotlib failed!")
try:
    import pydot
    print("Importing pydot successful")
except:
    print("Importing pydot failed!")
