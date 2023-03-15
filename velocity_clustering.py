

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
from pandas.plotting import scatter_matrix

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

from libs.auxfun import scale_bar
from libs.basemap import basemap
from libs.pyvelo import pyvelo


# configuration
period = 'post'
n_clusters = 8
Estimator = AgglomerativeClustering

# load the data
d4clus = pd.read_csv(f'data/ready4clustering_{period}.csv')

# some clustering algorithm in sklearn accepts weights
# so just in case I leave this here
std = np.sqrt(d4clus['elon']**2 + d4clus['elat']**2)
Cd = np.diag(std)
Wd = np.linalg.cholesky(Cd)

# input matrix
X = np.array(d4clus[['h', 'x_unit', 'y_unit']])

# scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# clustering
cls = Estimator(n_clusters)
y_pred = cls.fit_predict(X_scaled)
labels = np.unique(y_pred)

# save clusters
d4clus['cluster'] = y_pred
d4clus.to_csv(f'output/velocity/{period}_clusters.csv')


# ##############################################################################
# Plots
# ##############################################################################

cmap = get_cmap('gist_ncar')
colors = np.array([to_hex(cmap(i)) for i in np.arange(0,1,1/n_clusters)])

# params for the maps
scale = 0.1
paralelos = np.arange(-48, 17, 2)
meridianos = np.arange(-77, -55, 2)
extent = (-77, -55, -48, -17)
fmt = r'%2.0f$\degree$' # Format you want the ticks, e.g. '40%'
# pintar la tierra con un color plano
land = cfeature.NaturalEarthFeature(category='physical',
                            name='land', scale='10m',
                            edgecolor='face',
                            facecolor=cfeature.COLORS['land'])

# Figure 1
fig1, ax1 = plt.subplots(1, 1, figsize=(12,12),
                         subplot_kw={'projection': ccrs.PlateCarree()})
ax1.set_xticks(meridianos)
ax1.set_yticks(paralelos)
ax1.set_extent(extent)
ax1.tick_params(labelsize=13)

ticks = mticker.FormatStrFormatter(fmt)
ax1.xaxis.set_major_formatter(ticks)
ax1.yaxis.set_major_formatter(ticks)
ax1.add_feature(land, edgecolor='lightgray', alpha=1, zorder=0)
# agregar lineas de costa
ax1.coastlines(resolution='10m')

pyvelo(ax1, d4clus['lon'], d4clus['lat'], d4clus['vlon'], d4clus['vlat'],
       scale=scale)

#vel_scale = 20
#pyvelo(ax1, -59, -46, vel_scale, 0, scale=scale, arrow_prop={'width': 0.03})
#plt.text(-59, -46.8, f'{vel_scale} mm', fontsize=16)

scale_bar(ax1, 500, location=(0.8, 0.93))
fig1.savefig(f'output/velocity/{period}_map1.png', dpi=360, bbox_inches='tight')

#Figure 2
fig2, ax2 = plt.subplots(1,1, figsize=(15,10), subplot_kw={'projection':'polar'})
ax2.set_theta_zero_location("N")
ax2.set_theta_zero_location("N")
ax2.set_ylim(-0,40)
ax2.set_theta_direction(-1)
ax2.set_theta_direction(-1)
ax2.scatter(d4clus['azimut'], np.sqrt(d4clus['vlon']**2+d4clus['vlat']**2),
            color=colors[y_pred])
ax2.set_title('Azimut and V horizontal', fontsize=16)
fig2.savefig(f'output/velocity/{period}_polar.png', dpi=360, bbox_inches='tight')

# figure 3
fig3, ax3 = plt.subplots(1, 1, figsize=(15,5))
ax3.scatter(d4clus['vlon'], d4clus['vlat'], s=30, color=colors[y_pred])
ax3.set_xlabel('vlon')
ax3.set_ylabel('vlat')
fig3.savefig(f'output/velocity/{period}_xy.png', dpi=360, bbox_inches='tight')

# figure 4
fig5, ax5 = plt.subplots(1, 1, figsize=(12,12),
                         subplot_kw={'projection': ccrs.PlateCarree()})
ax5.scatter(d4clus['lon'], d4clus['lat'], s=40, color=colors[y_pred])
# especificar ubicación de los ticks
ax5.set_xticks(meridianos)
ax5.set_yticks(paralelos)
ax5.set_extent(extent)
ax5.tick_params(labelsize=13)

ax5.xaxis.set_major_formatter(ticks)
ax5.yaxis.set_major_formatter(ticks)

ax5.add_feature(land, edgecolor='lightgray', alpha=1, zorder=0)
# agregar lineas de costa
ax5.coastlines(resolution='10m')

scale_bar(ax5, 500, location=(0.8, 0.93))
fig5.savefig(f'output/velocity/{period}_map2.png', dpi=360, bbox_inches='tight')

#plt.show()
