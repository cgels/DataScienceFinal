import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
from numpy import genfromtxt

def get_data(filename):

    return genfromtxt(filename, delimiter=',')

def get_gr():
    gr = []
    for star in data[1:]:
        g = star[4]
        r = star[5]
        color = g - r
        gr.append(color)
    return gr

def get_id():
    id = []
    for star in data[1:]:
        id.append(star[0])
    return id

##########################################################################################
#Call functions here

data = get_data('stars.csv')
#Data format
#objid,ra,dec,u,g,r,i,z,distance
#id = 0
#ra = 1
#dec = 2
#u = 3
#g = 4
#r = 5
#i = 6
#z = 7
#distance = 8
print (data.shape)

gr = get_gr()
id = get_id()
print (gr)
print (id)