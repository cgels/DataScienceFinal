# DataScienceFinal

Hipparcos.csv

COLLUMS

HIP = Hipparcos star number
Vmag = Visual band magnitude.  This is an inverted logarithmic measure of brightness
RA = Right Ascension (degrees), positional coordinate in the sky equivalent to longitude on the Earth
DE = Declination (degrees), positional coordinate in the sky equivalent to latitude on the Earth
Plx = Parallactic angle (mas = milliarcsseconds).  1000/Plx gives the distance in parsecs (pc)
pmRA = Proper motion in RA (mas/yr).  RA component of the motion of the star across the sky
pmDE = Proper motion in DE (mas/yr). DE component of the motion of the star across the sky
e_Plx = Measurement error in Plx (mas)
B-V = Color of star (mag)

STATISTICAL QUESTIONS TO LOOK AT

Find Hyades cluster members, and possibly Hyades supercluster members, by multivariate clustering.
Validate the sample, and reproduce other results of Perryman et al. (1998)
Construct the HR diagram, and discriminate the main sequence and red giant branch in the full database and Hyades subset.  Can anything be learned about the `red clump' subgiants?
Isolate the Hyades main sequence and fit with nonparametric local regressions and with parametric regressions.
Use the heteroscedastic measurement error values e_Plx to weight the points in all of the above operations.
Can any unusual outliers be found? (white dwarfs, halo stars, runaway stars, ...)
