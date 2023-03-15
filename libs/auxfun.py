
from os import listdir
from os.path import isfile, join

import re

import cartopy.crs as ccrs
import numpy as np
import pandas as pd




def hillshade(array, azimuth, angle_altitude, dx=1):
    """
    Source: http://geoexamples.blogspot.com.br/2014/03/shaded-relief-images-using-gdal-python.html
    """
    x, y = np.gradient(array, dx)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi / 180.
    altituderad = angle_altitude*np.pi / 180.

    shaded = np.sin(altituderad)*np.sin(slope) + \
        np.cos(altituderad)*np.cos(slope)*np.cos(azimuthrad-aspect)

    return 255*(shaded + 1)/2


def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3,
              textsize=12):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length: 
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)        
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length) 

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth)
    #Plot the scalebar label
    ax.text(sbx, sby+5000, str(length) + ' km', transform=tmc,
            horizontalalignment='center', verticalalignment='bottom',
            fontsize=textsize)


def read_trench(trenchfile, id_trench='9921 CHILE TRENCH'):
    with open(trenchfile, 'r') as trench:
        content = trench.read().splitlines()
    trenches = [i for i, line in enumerate(content) if line[0]=='>']
    my_trenches = [i for i, line in enumerate(content) if line==id_trench]
    #mylines = [content[i+1] for i, line in enumerate(content) if i ]

    xy = []
    for line in mylines:
        lonlat = re.search('(-*\d+\.\d*)\s{1,5}(-*\d+\.\d*)', line)
        lonlat = (lonlat.group(1), lonlat.group(2))
        # lonlat = line.split(' ')
        lonlat = list(map(float, lonlat))
        xy.append(lonlat)
    xy = np.array(xy)

    return xy
