import numpy as np
import sklearn
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn
seaborn.set()
from numpy import genfromtxt



def get_data(filename):
    data = genfromtxt(filename, delimiter=',')
    data = [data[x] for x in range(1, data.shape[0])]
    data = np.asarray(data)
    return data



def get_bv():
    # bv = [data[x][8] for x in range(0, data.shape[0])]
    # bv = np.array(bv)
    return data[:,8]


def get_temp(color_index):
    # 9000 Kelvin / (B-V + .93)
    return 9000.0 / (color_index + 0.93)


def get_parallax():
    ## returns data for parallax - converting milliarcseconds to arcesonds
    return np.asarray(data[:,4]) / 1000


def get_id():
    # id = [data[x][0] for x in range(0, data.shape[0])]
    # id = np.array(data[:,0], dtype='int_')
    return np.array(data[:,0], dtype='int_')


def get_VMag():
    return np.array(data[:,1])

def get_RA():
    return np.asarray(data[:,2])

def get_dec():
    return np.asarray(data[:,3])

#L=(15 - Vmag - 5logPlx)/2.5
# Calculate the luminosity here
# NOTE: Solar Luminosity is equivalent to Absolute Magnitude.

def get_lum():
    Vmag = get_VMag()

    plx = get_parallax() * 1000
    plx = 5 * np.log10(plx)

    lum = (15.0 - Vmag - plx) / 2.5
    return np.asarray(lum)

## returns boolean mask for selecting features in dataset within an epsilon of the mean RA, Dec for the Hyades cluster
def get_diff_mean_Hyades_RA(ra, dec, epsilon=20):
    ## Hyades cluster is centered around a right ascension of 67 degrees
    print(ra)
    print(dec)
    ra_diff = ra - 67
    dec_diff = dec - 16

    ra_diff_within = ra_diff <= epsilon
    dec_diff_within = dec_diff <= epsilon

    mask = np.logical_and(ra_diff_within, dec_diff_within)
    return mask


## returns distance in parsec based on parallx angle
def get_dist(parallax):
    # constant in terms of astronmical units / parallax -> f
    print(parallax.size)
    return 206265.0 / parallax


def plot_hr(bv, lum, dist):
    plt.scatter(bv, lum, s=15, c='blue', marker='*', alpha=.7)
    plt.ylabel("Solar Luminosity (L☉)")
    plt.xlabel("Color Index: B-V (mag)")
    plt.show()

def plot_hr_hyades(bv, lum):
    plt.scatter(bv, lum, s=15, c='blue', marker='*', alpha=.7)
    plt.ylabel("Solar Luminosity (L☉)")
    plt.xlabel("Color Index: B-V (mag)")
    plt.show()



def plot_dist(ra, dec):
    plt.scatter(ra, dec, s=15, c='green', marker='*', alpha=0.7)
    plt.ylabel("Declination")
    plt.xlabel("Right ascension")
    plt.show()



def get_RA():
    return data[:, 2]



def get_DEC():
    return data[:, 3]


def lum_kmeans(bv, lum):
    plot = []

    for s in range(data.shape[0]):
        plot.append(np.array([bv[s], lum[s]]))

    plot = np.array(plot)

    # literally just took from notebook example, will mess around w params of KMeans later?
    clustering = KMeans(n_clusters=4, init='k-means++', n_init=5)
    clustering.fit(plot)
    clusters = clustering.predict(plot)

    plot_2Dclusters(clusters, bv, lum)

    return clusters




def dist_kmeans(ra, dec):
    # list of stars'[ra, dec] values
    plot = []

    for s in range(data.shape[0]):
        plot.append(np.array([ra[s], dec[s]]))

    plot = np.array(plot)

    # literally just took from notebook example, will mess around w params of KMeans later?
    clustering = KMeans(n_clusters=3, init='k-means++', n_init=5)
    clustering.fit(plot)
    clusters = clustering.predict(plot)

    return clusters



def plot_2Dclusters(clusters, x, y):
    cluster0 = clusters == 0
    cluster1 = clusters == 1
    cluster2 = clusters == 2

    plt.scatter(x[cluster0], y[cluster0], marker='D', c='blue')
    plt.scatter(x[cluster1], y[cluster1], marker='o', c='green')
    plt.scatter(x[cluster2], y[cluster2], marker='^', c='magenta')
    plt.xlabel("Right Ascension")
    plt.ylabel("Declination")
    plt.show()



def plot3D(x, y, z, clusters):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    cluster0 = clusters == 0
    cluster1 = clusters == 1
    cluster2 = clusters == 2

    ax.scatter(x[cluster0], y[cluster0], z[cluster0], marker='D', c='blue')
    ax.scatter(x[cluster1], y[cluster1], z[cluster1], marker='o', c='green')
    ax.scatter(x[cluster2], y[cluster2], z[cluster2], marker='^', c='magenta')
    ax.set_ylabel("Distance")
    ax.set_xlabel("Declination")
    ax.set_zlabel("Right Ascension")
    plt.show()


##########################################################################################
#Call functions here

data = get_data('HIPPARCOS.csv')
print (data.shape)
assert (data[0] == np.array([2,9.27,0.003797,-19.498837,21.9,181.21,-0.93,3.1,0.999])).sum() == 9
assert (data[-1] == np.array([118311,11.85,359.954685,-38.252603,24.63,337.76,-112.81,2.96,1.391])).sum() == 9

bv = get_bv()
# print (bv)

id = get_id()
# print (id)

lum = get_lum()
# print (lum)

ra = get_RA()

dec = get_DEC()

temp = get_temp(bv)

parallax = get_parallax()
# print(parallax)

dist = get_dist(parallax)

hyades_mask = get_diff_mean_Hyades_RA(get_RA(), get_dec(), 1)

# print(dist)
plot_hr(bv, lum, dist)
#
# plot_dist(ra, dec)
#plot_hr(bv, lum, dist)

plot_hr_hyades(bv[hyades_mask], lum[hyades_mask])


distClusters = dist_kmeans(ra, dec)
# print(distClusters)
# plot_2Dclusters(distClusters, ra, dec)
plot3D(dec, dist, ra, distClusters)