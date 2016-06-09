import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
import data
import visuals
import hyades_analysis
import seaborn
seaborn.set()


## returns boolean mask for selecting features in data within an epsilon of the mean RA, Dec for the Hyades cluster
def get_diff_mean_Hyades_RA(ra, dec, epsilon=20):
    ## Hyades cluster is centered around a right ascension of 67 degrees
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
    # constant in terms of astronomical units / parallax -> f
    # print(parallax.size)
    return 206265.0 / parallax





def lum_kmeans(bv, lum):
    plot = []

    for s in range(dataset.shape[0]):
        plot.append(np.array([bv[s], lum[s]]))

    plot = np.array(plot)

    # literally just took from notebook example, will mess around w params of KMeans later?
    clustering = KMeans(n_clusters=4, init='k-means++', n_init=10)
    clustering.fit(plot)
    clusters = clustering.predict(plot)

    visuals.plot_2Dclusters(clusters, bv, lum)

    return clusters


def proper_motion_kmeans():
    plot = []


    ra = data.get_ra(dataset)
    dec = data.get_dec(dataset)

    for s in range(dataset.shape[0]):
        plot.append(np.array([pm_ra[s], pm_dec[s], ra[s], dec[s]]))

    clustering = KMeans(n_clusters=2, init='k-means++', n_init=10)
    clustering.fit(plot)
    clusters = clustering.predict(plot)

    visuals.plot_2Dclusters(clusters, bv, lum, "Clustering by k-Means of Proper Motion")

    return clusters


def dist_kmeans(ra, dec):
    # list of stars'[ra, dec] values
    plot = []
    for s in range(dataset.shape[0]):
        plot.append(np.array([ra[s], dec[s]]))

    plot = np.array(plot)

    # literally just took from notebook example, will mess around w params of KMeans later?
    clustering = KMeans(n_clusters=3, init='k-means++', n_init=10)
    clustering.fit(plot)
    clusters = clustering.predict(plot)
    return clusters

def dist_spectral(x, y):

    plot = []
    for s in range(dataset.shape[0]):
        plot.append(np.array([x[s], y[s]]))
    plot = np.array(plot)
    spectral = SpectralClustering(n_clusters=3, eigen_solver='arpack', affinity="nearest_neighbors")
    clusters = spectral.fit_predict(plot)
    return clusters

def dist_dbscan(x, y, z):
    plot = []
    for s in range(dataset.shape[0]):
        plot.append(np.array([x[s], y[s]], z[s]))
    plot = np.array(plot)
    db = DBSCAN(eps=.5, min_samples=3, metric='euclidean').fit(plot)
    db = db.labels_
    return db


def galactic_dist_kmeans():

    plot = []
    for s in range(dataset.shape[0]):
        plot.append(np.array([l[s], b[s]]))

    plot = np.array(plot)

    clustering = KMeans(n_clusters=5, init='k-means++', n_init=10)
    clustering.fit(plot)
    clusters = clustering.predict(plot)

    return clusters





##########################################################################################
# Call functions here

dataset = data.get_data("HIPPARCOS.csv")
print (dataset.shape)
assert (dataset[0] == np.array([2, 9.27, 0.003797, -19.498837, 21.9, 181.21, -0.93, 3.1, 0.999])).sum() == 9
assert (dataset[-1] == np.array([118311, 11.85, 359.954685, -38.252603, 24.63, 337.76, -112.81, 2.96, 1.391])).sum() == 9

hyades = data.get_hyades(dataset)
hyadesVector = data.hyades_vector(dataset, hyades) == 1

id = data.get_id(dataset)
bv = data.get_bv(dataset)
lum = data.get_lum(dataset)
ra = data.get_ra(dataset)
dec = data.get_dec(dataset)
pm_ra = data.get_pmRA(dataset)
pm_dec = data.get_pmDec(dataset)
temp = data.get_temp(bv)
parallax = data.get_parallax(dataset)
dist = get_dist(parallax)
l = data.get_galactic_latitude(dataset)
b = data.get_galactic_longitude(dataset)

# hyades_pm = get_Hyades_proper_motion()

hyades_mask = get_diff_mean_Hyades_RA(ra, dec, 1)

hyades_mask_plx = get_Hyades_mean_parallax(parallax, .01)
#
# plot_dist(ra, dec)
# plot_hr(bv, lum)
# visuals.plot_2Dclusters(hyadesVector, bv, lum, "Ground Truth Hyades")
# print("Hyades By RA, DEC - # Data Points: {}".format(np.sum(hyades_mask)))
# print("Hyades By RA, DEC Accuracy: {}".format(np.sum(np.logical_and(hyades_mask, hyadesVector)) / np.sum(hyadesVector)))
# # plot_hr_hyades(bv[hyades_mask], lum[hyades_mask])
# visuals.plot_2Dclusters(hyades_mask, bv, lum, "Hyades Clustered by RA and DEC")
# print("Hyades By Parallax - # Data Points: {} ".format(np.sum(hyades_mask_plx)))
# print("Hyades By Parallax Accuracy: {}".format(
#     np.sum(np.logical_and(hyades_mask_plx, hyadesVector)) / np.sum(hyadesVector)))
# # plot_hr_hyades_plx(bv[hyades_mask_plx], lum[hyades_mask_plx])
# visuals.plot_2Dclusters(hyades_mask_plx, bv, lum, "Hyades Clustered by Parallax")
# print("Hyades By RA, DEC && Parallax - # Data Points: {}".format(np.sum(np.logical_and(hyades_mask_plx, hyades_mask))))
# print("Hyades By RA, DEC && Parallax Accuracy: {}".format(
#     np.sum(np.logical_and(np.logical_and(hyades_mask_plx, hyades_mask), hyadesVector)) / np.sum(hyadesVector)))
# # plot_hr_hyades_plx_AND_ra_dec(bv[np.logical_and(hyades_mask_plx, hyades_mask)], lum[np.logical_and(hyades_mask_plx, hyades_mask)])
# visuals.plot_2Dclusters(np.logical_and(hyades_mask_plx, hyades_mask), bv, lum, "Hyades Clustered by RA, DEC, and Parallax")

# propMotClusters = proper_motion_kmeans()

# distClusters = dist_kmeans(ra, dec)
#
# spectral_mask = dist_spectral(ra, dec)
# # spectral_mask_plx = dist_spectral_plx(parallax, .01)
# print("Hyades By RA, DEC Spectral - # Data Points: {}".format(np.sum(spectral_mask)))
# print("Hyades By Spectral Accuracy: {}".format(np.sum(np.logical_and(spectral_mask, hyadesVector))/np.sum(hyadesVector)))
# visuals.plot_2Dclusters(spectral_mask, bv, lum, "Hyades Clustered by RA, DEC, and Parallax - Spectral")
# accuracy = visuals.plot_with_hyades(hyadesVector, spectral_mask, bv, lum, "Hyades Clustered by RA, DEC, and Parallax - Spectral")
# print(accuracy)
# db_mask = dist_dbscan(ra, dec, parallax)
# print("Hyades By RA, DEC Parallax - # Data Points: {}".format(np.sum(db_mask)))
# print("Hyades By DBSCAN Accuracy: {}".format(np.sum(np.logical_and(db_mask, hyadesVector))/np.sum(hyadesVector)))
# # visuals.plot_2Dclusters(db_mask, bv, lum, "Hyades Clustered by RA, DEC, and Parallax - DBSCAN")
# # acc = visuals.plot_with_hyades(hyadesVector, db_mask, bv, lum, "Hyades Clustered by RA, DEC, and Parallax - DBSCAN")
# # visuals.plot3D(dec, parallax, ra, db_mask)
# # print(distClusters)
# # plot_2Dclusters(distClusters, ra, dec)
# # plot3D(dec, dist, ra, distClusters)
#
# print("GALACTIC")
# galDistClusters = galactic_dist_kmeans()
# visuals.plot_2Dclusters(galDistClusters, bv, lum, "Clustering with Galactic Coordinate")
# # print(distClusters)
# # plot_2Dclusters(distClusters, ra, dec)
# visuals.plot3D(dec, dist, ra, distClusters)


print("KMEANS Plotted results for clustering for each criterion for k 1 - 15.")
vecs = [ra, dec, pm_ra, pm_dec, parallax, dist, l, b]
hyades_study = hyades_analysis.find_optimal_kmeans(15, vecs, hyadesVector)
titles = []
titles.append("Right Ascension, Declination [deg, deg]")
titles.append("Parallax [milli-arcseconds (mas)]")
titles.append("Distance [parsec (pc)]")
titles.append("Galactic Longitude, Latitude [deg, deg]")
titles.append("Proper Motions in Right Ascension, Declination  [mas/yr, mas/yr]")
titles.append("Right Ascension, Declination, Distance [deg, deg, pc]")
titles.append("Right Ascension, Declination, Parallax [deg, deg, mas]")
titles.append("Distance, Longitude / Latitude [pc, deg, deg]")
titles.append("Distance, Proper Motions in Right Ascension, Declination [pc, mas/yr, mas/yr]")
# hyades_study = hyades_analysis.find_optimal_kmeans(15, vecs, hyadesVector)
hyades_study = hyades_analysis.find_optimal_kmeans(15, vecs, hyadesVector)
kmeans_prefix = "kMeans Clustering on "
suffix = "\n {:.1f}% of Hyades Cluster correctly identified"
for s in range(len(hyades_study)):
    clusters = hyades_study[s][1]
    print("Best Accuracy for Study {} for k = {} is {}".format(s, hyades_study[s][2], hyades_study[s][0]))
    ## pass a list of clusters that we want to examine -- best cluster per study is available at
    visuals.plot_best_clust_with_hyades(hyadesVector, clusters, bv, lum, kmeans_prefix + titles[s] + suffix, hyades_study[s][3], hyades_study[s][0])
# print()
# print("DBSCAN Plotted results for clustering for each criterion for 15 Epsilons = [.01 - 2.5]")
# hyades_study = hyades_analysis.find_optimal_dbscan(10, vecs, hyadesVector)
#
# for s in range(len(hyades_study)):
#     clusters = hyades_study[s][1]
#     print("Best Accuracy for Study {} for epsilon = {:.3f} is {}".format(s, hyades_study[s][2], hyades_study[s][0]))
#     ## pass a list of clusters that we want to examine -- best cluster per study is available at
#     visuals.plot_best_clust_with_hyades(hyadesVector, clusters, bv, lum,, titles[s], hyades_study[s][3])

#
# print()
# print("SPECTRAL Plotted results for clustering for k = 1 - 15")
# hyades_study = hyades_analysis.find_optimal_spectral_clusters(15, vecs, hyadesVector)
#
# for s in range(len(hyades_study)):
#     clusters = hyades_study[s][1]
#     if clusters != None:
#         print("Best Accuracy for Study {} for k = {} is {}".format(s, hyades_study[s][2], hyades_study[s][0]))
#         ## pass a list of clusters that we want to examine -- best cluster per study is available at
#         visuals.plot_best_clust_with_hyades(hyadesVector, clusters, bv, lum, , titles[s], hyades_study[s][3])
#     else:
#         print("Study {} ommitted.".format(s))