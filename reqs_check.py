# -*- coding: utf-8 -*-

import sys

check = True

python_version = list(sys.version_info)
if python_version[0] != 3 or (python_version[0] == 3 and python_version[1] < 7):
    print("Your Python version does not meet requirements. (Python >3.7)")
    check = False
    exit(1)
else:
    print("Your Python version satisfies the requirements.")
try:
    import keras
    print("Importing keras successful")
except:
    print("Importing keras failed!")
    check = False
try:
    import tensorflow
    print("Importing tensorflow successful")
except:
    print("Importing tensorflow failed!")
    check = False
try:
    import numpy
    print("Importing numpy successful")
except:
    print("Importing numpy failed!")
    check = False
try:
    import pandas
    print("Importing pandas successful")
except:
    print("Importing pandas failed!")
    check = False
try:
    import pydot
    print("Importing pydot successful")
except:
    print("Importing pydot failed!")
    check = False
if check:
    print("All dependencies are install successfully!")
else:
    print("Some dependencies are not install successfully. Please reinstall it or create new environment.")
