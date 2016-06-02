import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
from numpy import genfromtxt

def get_data(filename):
    data = genfromtxt(filename, delimiter=',')
    data = [data[x] for x in range(1, data.shape[0])]
    data = np.asarray(data)
    return data



def get_gr():
    gr = []
    for star in data[1:]:
        g = star[4]
        r = star[5]
        color = g - r
        gr.append(color)
    return gr



def get_id():
    id = [data[x][0] for x in range(1, data.shape[0])]
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