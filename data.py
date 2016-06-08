import numpy as np
from numpy import genfromtxt

def get_data(filename):
    data = genfromtxt(filename, delimiter=',')
    data = [data[x] for x in range(1, data.shape[0])]
    data = np.asarray(data)
    return data


def get_hyades(data):
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


def hyades_vector(data, hyades):
    hyadesCluster = []
    for x in range(data.shape[0]):
        if data[x][0] in hyades.keys():
            hyadesCluster.append(hyades[data[x][0]])
        else:
            hyadesCluster.append(0)

    return np.array(hyadesCluster)


def get_bv(data):
    ## get color index vector
    return data[:,8]


def get_temp(color_index):
    # 9000 Kelvin / (B-V + .93)
    return 9000.0 / (color_index + 0.93)


def get_id(data):
    ## get hipparcos star id
    return np.array(data[:,0], dtype='int_')



def get_VMag(data):
    # get visual magnitude vector
    return np.array(data[:,1])


def get_ra(data):
    ## get right ascension vector (celestial coordinates)
    return np.array(data[:,2])


def get_dec(data):
    ## get declination (celestial coordinates)
    return np.array(data[:,3])


def get_parallax(data):
    ## get parallax vector in arc seconds
    return np.array(data[:,4]) / 1000


def get_pmRA(data):
    ## get proper motion right ascension vector
    return np.array(data[:,6])


def get_pmDec(data):
    ## get proper motion declination vector
    return np.array(data[:,7])


def get_galactic_latitude(data):
    alpha = get_ra(data)
    delta = get_dec(data)
    C1 = 192.25 * 3600
    C2 = 27.4 * 3600

    num = np.sin(C1 - alpha)
    denom = np.cos(C1 - alpha) * np.sin(C2) - np.tan(delta) * np.cos(C2)

    return np.arctan( num / denom )


def get_galactic_longitude(data):
    alpha = get_ra(data)
    delta = get_dec(data)
    C1 = 192.25 * 3600
    C2 = 27.4 * 3600

    x = np.sin(delta) * np.sin(C2)
    y = np.cos(delta) * np.cos(C2) * np.cos(C1 - alpha)

    return x + y


# L=(15 - Vmag - 5logPlx)/2.5
# Calculate the luminosity here
# NOTE: Solar Luminosity is equivalent to Absolute Magnitude.
def get_lum(data):
    Vmag = get_VMag(data)

    plx = get_parallax(data) * 1000
    plx = 5 * np.log10(plx)

    lum = (15.0 - Vmag - plx) / 2.5
    return np.array(lum)