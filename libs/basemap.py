"""
Prof. Francisco Hernán Ortega Culaciati
ortega.francisco@u.uchile.cl
Departamento de Geofísica - FCFM
Universidad de Chile

May 8, 2018

Provides a function to create a basemap using cartopy that can be used to 
plot other stuff in it. 

Right now only works with PlateCarree map projection. 

"""
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from collections.abc import Iterable

def basemap(figNumber, lonmin, lonmax, latmin, latmax, figsize = [8,7],
            projection = 'PlateCarree', cLON = None, cLAT = None,
            addLand = True, resolution = '10m',
            coastline_color = 'lightgray', coastlinewidth = 1.0,
            xticker = 1.0, yticker = 1.0, fig = None, ax = None,
            xlabels = True, ylabels = True, 
            xlabels_top = False, ylabels_right = False, 
            tick_label_size = 12,
            addOcean = False):
    """

    :param figNumber:
    :param lonmin:
    :param lonmax:
    :param latmin:
    :param latmax:
    :param figsize:
    :param projection:
    :param addLand:
    :param resolution:
    :return:
    """
    # zorder is related to layer ordering when plotting in matplotlib
    zorder = 0

    # if the center of the map projection is not given take the center of the map
    # region
    if cLON is None:
        cLON = 0.5 * (lonmin + lonmax)
    if cLAT is None:
        cLAT = 0.5 * (latmin + latmax)

    if projection == 'PlateCarree':
        mapproj = ccrs.PlateCarree()
    else:
        raise NotImplementedError('Only PlateCarree map projection is implemented...')

    # initialize figure and axis
    if fig is None or ax is None:
        fig = plt.figure(figNumber)
        fig.set_size_inches(*figsize, forward=True)
        ax = plt.axes(projection=mapproj)


    # set map extent
    ax.set_extent([lonmin, lonmax, latmin, latmax])
    ax.set_adjustable('datalim')
    ax.set_aspect('equal')

    # add land
    if addLand:
        land = cfeature.NaturalEarthFeature(category='physical', name='land',
                                            scale=resolution,
                                            edgecolor='face',
                                            facecolor=cfeature.COLORS['land'])
        zorder = 0
        ax.add_feature(land, edgecolor=coastline_color, alpha=0.5, zorder=zorder)

    zorder = 1
    ax.coastlines(resolution=resolution, color=coastline_color,
                  linewidth = coastlinewidth,
                  zorder=zorder)
    zorder = 2

    # add ocean
    if addOcean:
        ocean = cfeature.NaturalEarthFeature(category = 'physical', 
                                             name = 'ocean', 
                                             scale = resolution, 
                                             edgecolor='face',
                                             facecolor='b')
        zorder = 0
        ax.add_feature(ocean, edgecolor=coastline_color, alpha=0.5, zorder=zorder)

    # define tick locations
    if isinstance(xticker, Iterable):
        xlocator = mticker.FixedLocator(xticker, nbins=None)
    else:
        xlocator = mticker.MultipleLocator(xticker)

    if isinstance(yticker, Iterable):
        ylocator = mticker.FixedLocator(yticker, nbins=None)
    else:
        ylocator = mticker.MultipleLocator(yticker)
  

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.8, linestyle='--',
                      zorder=zorder)

    #gl.xlabels_bottom = False
    gl.xlabels_top = xlabels_top

    gl.ylabels_right = ylabels_right

    gl.xlabels_bottom = xlabels
    gl.ylabels_left = ylabels

    gl.xlines = True
    gl.ylines = True

    # set x ticks and y ticks at multiples of xticker and yticker respectiverly
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
  
    gl.xlabel_style = {'size': tick_label_size, 'color': 'k'}
    gl.ylabel_style = {'size': tick_label_size, 'color': 'k'}

    gl.xlocator = xlocator
    gl.ylocator = ylocator

  
    return fig, ax, zorder
