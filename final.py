import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
from numpy import genfromtxt

def get_data(filename):

    return genfromtxt(filename, delimiter=',')


data = get_data('result.csv')