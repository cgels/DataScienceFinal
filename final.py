import numpy as np
import sklearn
from sklearn.cluster import KMeans
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



def get_bv():

    bv = [data[x][8] for x in range(0, data.shape[0])]
    bv = np.array(bv)
    return bv



def get_id():
    id = [data[x][0] for x in range(0, data.shape[0])]
    id = np.array(id,dtype='int_')
    return id



#L=(15 - Vmag - 5logPlx)/2.5
# Calculate the luminosity here

def get_lum():

    Vmag = [data[x][1] for x in range(0, data.shape[0])]
    Vmag = np.array(Vmag)

    plx = [data[x][4] for x in range(0, data.shape[0])]
    plx = np.array(plx)

    plx = np.log10(plx)
    plx = plx * 5

    lum = 15 - Vmag - plx
    lum = lum/2.5
    lum = np.array(lum)

    return 10**lum




def plot_hr(bv, lum):
    lum = np.log10(lum)
    plt.scatter(bv, lum, s=20, c='blue', marker='*')
    plt.ylabel("Luminosity (L☉)")
    plt.xlabel("B-V (mag)")
    plt.show()



def kmeans(bv, lum):
    # list of stars'(bv, lum) values
    plot = []

    for s in range(data.shape[0]):
        plot.append((bv[s], lum[s]))

    plot = np.array(plot)
    print(plot)

    # literally just took from notebook example, will mess around w params of KMeans later?
    clustering = KMeans(n_clusters=3, init='random', n_init=5)
    clustering.fit(plot)
    clusters = clustering.predict(plot)
    print(clusters)
    return clusters


##########################################################################################
#Call functions here

data = get_data('HIPPARCOS.csv')
print (data.shape)
assert (data[0] == np.array([2,9.27,0.003797,-19.498837,21.9,181.21,-0.93,3.1,0.999])).sum() == 9
assert (data[-1] == np.array([118311,11.85,359.954685,-38.252603,24.63,337.76,-112.81,2.96,1.391])).sum() == 9

bv = get_bv()
print (bv)

id = get_id()
print (id)

lum = get_lum()
print (lum)

#plot_hr(bv, lum)

kmeans(bv, lum)