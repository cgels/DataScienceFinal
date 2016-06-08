import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as plt_clr
import itertools
import seaborn

seaborn.set()
from mpl_toolkits.mplot3d import Axes3D


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


def get_plot_masks(clusters):
    masks = {c: None for c in np.unique(clusters)}
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


def plot_with_hyades(hv, clusters, x, y, title):
    masks, marker, clr_lst = get_plot_masks(clusters)

    i = 0
    acc_arr = []
    for c in masks:
        acc = float(np.sum(np.logical_and(masks[c], hv))) / np.sum(hv)
        print(np.sum(np.logical_and(masks[c], hv)))
        print(np.sum(hv))
        print(i, "  ", acc)
        acc_arr.append(acc)
        plt.scatter(x[hv], y[hv], marker=marker.__next__(), c=clr_lst.__next__())
        plt.scatter(x[masks[c]], y[masks[c]], marker=marker.__next__(), c=clr_lst.__next__())
        plt.title(title)
        plt.ylabel("Solar Luminosity (L☉)")
        plt.xlabel("Color Index: B-V (mag)")
        plt.show()
        i += 1

    acc_arr = np.array(acc_arr)
    return acc_arr


def plot_best_clust_with_hyades(hv, clusters, x, y, title, best_cluster_label):
    masks, marker, clr_lst = get_plot_masks(clusters)

    acc_arr = []
    hits = float(np.sum(np.logical_and(masks[best_cluster_label], hv)))
    acc = hits / np.sum(hv)
    acc_arr.append(acc)

    plt.scatter(x[hv], y[hv], marker=marker.__next__(), c=clr_lst.__next__())
    plt.scatter(x[masks[best_cluster_label]], y[masks[best_cluster_label]], marker=marker.__next__(),
                c=clr_lst.__next__())

    plt.title(title)
    plt.ylabel("Solar Luminosity (L☉) for Cluster {}".format(best_cluster_label))
    plt.xlabel("Color Index: B-V (mag) for Cluster {}".format(best_cluster_label))
    plt.show()

    acc_arr = np.array(acc_arr)
    return acc_arr


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
