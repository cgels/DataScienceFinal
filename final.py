import numpy as np
import sklearn
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as plt_clr
from mpl_toolkits.mplot3d import Axes3D
import itertools
import seaborn
seaborn.set()
from numpy import genfromtxt


def get_data(filename):
    data = genfromtxt(filename, delimiter=',')
    data = [data[x] for x in range(1, data.shape[0])]
    data = np.asarray(data)
    return data


def get_hyades():
    hyades = []
    hyadesCluster = {}

    file = open("HyadesCluster.csv", "r")
    for line in file:
        newLine = line.replace("\n", "")
        star = newLine.replace("?", "-1").split(",")
        hyades.append(star)

        if star[0] not in hyadesCluster.keys():
            hyadesCluster[int(star[0])] = int(star[1])

    return hyadesCluster


def hyades_vector():
    hyadesCluster = []
    for x in range(data.shape[0]):
        if data[x][0] in hyades.keys():
            hyadesCluster.append(hyades[data[x][0]])
        else:
            hyadesCluster.append(0)

    return np.array(hyadesCluster)


def get_bv():
    ## get color index vector
    return data[:,8]


def get_temp(color_index):
    # 9000 Kelvin / (B-V + .93)
    return 9000.0 / (color_index + 0.93)


def get_id():
    ## get hipparcos star id
    return np.array(data[:,0], dtype='int_')



def get_VMag():
    # get visual magnitude vector
    return np.array(data[:,1])


def get_ra():
    ## get right ascension vector (celestial coordinates)
    return np.asarray(data[:,2])


def get_dec():
    ## get declination (celestial coordinates)
    return np.asarray(data[:,3])


def get_parallax():
    ## get parallax vector in arc seconds
    return np.asarray(data[:,4]) / 1000


def get_pmRA():
    ## get proper motion right ascension vector
    return np.asarray(data[:,6])


def get_pmDec():
    ## get proper motion declination vector
    return np.asarray(data[:,7])


def get_galactic_latitude():
    alpha = get_ra()
    delta = get_dec()
    


# L=(15 - Vmag - 5logPlx)/2.5
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
    # print(ra)
    # print(dec)
    ra_diff = ra - 67
    dec_diff = dec - 16

    ra_diff_within = ra_diff <= epsilon
    dec_diff_within = dec_diff <= epsilon

    mask = np.logical_and(ra_diff_within, dec_diff_within)
    return mask



def get_Hyades_mean_parallax(plx, epsilon=20):
    mean_plx = 22.0
    mean_dif_sqr = (plx * 1000) - mean_plx
    return mean_dif_sqr <= epsilon



def get_Hyades_proper_motion():
    propMot = proper_motion_kmeans()
    return propMot


## returns distance in parsec based on parallx angle
def get_dist(parallax):
    # constant in terms of astronmical units / parallax -> f
    # print(parallax.size)
    return 206265.0 / parallax



def plot_hr(bv, lum):
    clusters = spectral_knn(bv)
    B = clusters == 0
    A = clusters == 1
    F = clusters == 2
    G = clusters == 3
    K = clusters == 4
    M = clusters == 5

    # creating figure instance
    fig = plt.figure(1)
    rect = fig.patch
    rect.set_facecolor('black')
    fig.savefig('whatever.png', facecolor=fig.get_facecolor(), edgecolor='none')

    plt.scatter(bv[B], lum[B], marker='*', c='lightblue')
    plt.scatter(bv[A], lum[A], marker='*', c='white')
    plt.scatter(bv[F], lum[F], marker='*', c='lightyellow')
    plt.scatter(bv[G], lum[G], marker='*', c='yellow')
    plt.scatter(bv[K], lum[K], marker='*', c='orange')
    plt.scatter(bv[M], lum[M], marker='*', c='red')
    plt.ylabel("Solar Luminosity (L☉)")
    plt.xlabel("Color Index: B-V (mag)")

    plt.show()


def plot_hr_hyades(bv, lum):
    plt.scatter(bv, lum, s=15, c='blue', marker='*', alpha=.7)
    plt.ylabel("Solar Luminosity (L☉)")
    plt.xlabel("Color Index: B-V (mag)")
    plt.show()


def plot_hr_hyades_plx(bv, lum):
    plt.scatter(bv, lum, s=15, c='blue', marker='*', alpha=.7)
    plt.ylabel("Solar Luminosity (L☉)")
    plt.xlabel("Color Index: B-V (mag)")
    plt.show()


def plot_hr_hyades_plx_AND_ra_dec(bv, lum):
    plt.scatter(bv, lum, s=15, c='blue', marker='*', alpha=.7)
    plt.ylabel("Solar Luminosity (L☉)")
    plt.xlabel("Color Index: B-V (mag)")
    plt.show()


def plot_dist(ra, dec):
    plt.scatter(ra, dec, s=15, c='green', marker='*', alpha=0.7)
    plt.ylabel("Declination")
    plt.xlabel("Right ascension")
    plt.show()


def spectral_knn(bv):
    mask = []
    for point in bv:
        if point <= -0.04:
            mask.append(0)
        elif point > -0.04 and point <= 0.3:
            mask.append(1)
        elif point > 0.3 and point <= 0.53:
            mask.append(2)
        elif point > 0.53 and point <= 0.74:
            mask.append(3)
        elif point > 0.74 and point <= 1.33:
            mask.append(4)
        elif point > 1.33:
            mask.append(5)
        else:
            mask.append(-1)

    mask = np.array(mask)
    assert mask.shape[0] == bv.shape[0]
    return mask


def lum_kmeans(bv, lum):
    plot = []

    for s in range(data.shape[0]):
        plot.append(np.array([bv[s], lum[s]]))

    plot = np.array(plot)

    # literally just took from notebook example, will mess around w params of KMeans later?
    clustering = KMeans(n_clusters=4, init='k-means++', n_init=10)
    clustering.fit(plot)
    clusters = clustering.predict(plot)

    plot_2Dclusters(clusters, bv, lum)

    return clusters


def proper_motion_kmeans():
    plot = []
    pm_ra = get_pmRA()
    pm_dec = get_pmDec()
    ra = get_ra()
    dec = get_dec()

    for s in range(data.shape[0]):
        plot.append(np.array([pm_ra[s], pm_dec[s], ra[s], dec[s]]))

    clustering = KMeans(n_clusters=2, init='k-means++', n_init=10)
    clustering.fit(plot)
    clusters = clustering.predict(plot)

    plot_2Dclusters(clusters, get_bv(), get_lum(), "Clustering by k-Means of Proper Motion")

    return clusters


def dist_kmeans(ra, dec):
    # list of stars'[ra, dec] values
    plot = []
    for s in range(data.shape[0]):
        plot.append(np.array([ra[s], dec[s]]))

    plot = np.array(plot)

    # literally just took from notebook example, will mess around w params of KMeans later?
    clustering = KMeans(n_clusters=3, init='k-means++', n_init=10)
    clustering.fit(plot)
    clusters = clustering.predict(plot)

    return clusters


def get_plot_masks(clusters):
    masks = { c : None for c in np.unique(clusters) }
    clr_lst = itertools.cycle(tuple(plt_clr.cnames.keys()))
    marker = itertools.cycle(('*', 's', '^', 'o', 'D'))

    for c in masks:
        masks[c] = clusters == c
    return masks, marker, clr_lst

def plot_2Dclusters(clusters, x, y, title):
    masks, marker, clr_lst = get_plot_masks(clusters)

    for c in masks:
        plt.scatter(x[masks[c]], y[masks[c]], marker=marker.__next__(), c=clr_lst.__next__())

    plt.title(title)
    plt.ylabel("Solar Luminosity (L☉)")
    plt.xlabel("Color Index: B-V (mag)")
    plt.show()


def plot3D(x, y, z, clusters):
    masks, markerMap, clrMap = get_plot_masks(clusters)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for c in masks:
        ax.scatter(x[masks[c]], y[masks[c]], z[masks[c]], marker=markerMap.__next__(), c=clrMap.__next__(), alpha=.7)

    ax.set_ylabel("Distance")
    ax.set_xlabel("Declination")
    ax.set_zlabel("Right Ascension")
    plt.show()


##########################################################################################
# Call functions here

data = get_data('HIPPARCOS.csv')
# print (data.shape)
assert (data[0] == np.array([2, 9.27, 0.003797, -19.498837, 21.9, 181.21, -0.93, 3.1, 0.999])).sum() == 9
assert (data[-1] == np.array([118311, 11.85, 359.954685, -38.252603, 24.63, 337.76, -112.81, 2.96, 1.391])).sum() == 9

hyades = get_hyades()
hyadesVector = hyades_vector() == 1

bv = get_bv()
# print (bv)

id = get_id()
bv = get_bv()
lum = get_lum()
ra = get_ra()
dec = get_dec()
temp = get_temp(bv)
parallax = get_parallax()
dist = get_dist(parallax)

hyades_pm = get_Hyades_proper_motion()

hyades_mask = get_diff_mean_Hyades_RA(get_ra(), get_dec(), 1)

hyades_mask_plx = get_Hyades_mean_parallax(parallax, .01)
#
# plot_dist(ra, dec)
# plot_hr(bv, lum)
plot_2Dclusters(hyadesVector, bv, lum, "Ground Truth Hyades")
print("Hyades By RA, DEC - # Data Points: {}".format(np.sum(hyades_mask)))
print("Hyades By RA, DEC Accuracy: {}".format(np.sum(np.logical_and(hyades_mask, hyadesVector))/np.sum(hyadesVector)))
# plot_hr_hyades(bv[hyades_mask], lum[hyades_mask])
plot_2Dclusters(hyades_mask, bv, lum, "Hyades Clustered by RA and DEC")
print("Hyades By Parallax - # Data Points: {} ".format(np.sum(hyades_mask_plx)))
print("Hyades By Parallax Accuracy: {}".format(np.sum(np.logical_and(hyades_mask_plx, hyadesVector))/np.sum(hyadesVector)))
# plot_hr_hyades_plx(bv[hyades_mask_plx], lum[hyades_mask_plx])
plot_2Dclusters(hyades_mask_plx, bv, lum, "Hyades Clustered by Parallax")
print("Hyades By RA, DEC && Parallax - # Data Points: {}".format(np.sum(np.logical_and(hyades_mask_plx, hyades_mask))))
print("Hyades By RA, DEC && Parallax Accuracy: {}".format(np.sum(np.logical_and(np.logical_and(hyades_mask_plx, hyades_mask), hyadesVector))/np.sum(hyadesVector)))
# plot_hr_hyades_plx_AND_ra_dec(bv[np.logical_and(hyades_mask_plx, hyades_mask)], lum[np.logical_and(hyades_mask_plx, hyades_mask)])
plot_2Dclusters(np.logical_and(hyades_mask_plx, hyades_mask), bv, lum, "Hyades Clustered by RA, DEC, and Parallax")

# propMotClusters = proper_motion_kmeans()

distClusters = dist_kmeans(ra, dec)
# print(distClusters)
# plot_2Dclusters(distClusters, ra, dec)
plot3D(dec, dist, ra, distClusters)
