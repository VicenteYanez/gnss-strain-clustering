

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
from pandas.plotting import scatter_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from libs.basemap import basemap
from libs.pyvelo import pyvelo
from libs.auxfun import scale_bar


n_clusters = 9  #12 #11 #20    #7#6#11#10
Estimator = AgglomerativeClustering
period = 'post'
extent = (-77, -57, -48, -17)  # map extent
paralelos = np.arange(-48, -17, 2)
meridianos = np.arange(-77, -57, 2)

# load data
d4clus = pd.read_csv(f'data/ready4clustering_{period}.csv')

# features to clusterize
# no funciona esto X = np.array(d4clus[['S_inv1', 'S_inv3', 'wz','t1','t2']])
X = np.array(d4clus[['S_inv1', 'S_inv3', 'wz']])
#X = np.array(d4clus[['S_inv1', 'S_inv3', 'wz', 'evalue1', 'evalue2','s_max']])
  # since S_inv3 is so small that the scaler sometimes could not work

# scaling, or MinMaxScaler depending of the stats of the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cls = Estimator(n_clusters)
y_pred = cls.fit_predict(X_scaled)
labels = np.unique(y_pred)

# save clusters
d4clus['cluster'] = y_pred
d4clus.to_csv(f'output/strain_rot/{period}_clusters_{n_clusters}.csv')

# ##############################################################################
# Plots
# ##############################################################################

# please be careful for too similar colors when testing for many clusters!!!!
cmap = get_cmap('tab20b')
colors = np.array([to_hex(cmap(i)) for i in np.arange(0,1,1/n_clusters)])

# Figure 2
fig2, ax2 = plt.subplots(1,1, figsize=(15,10),
                         subplot_kw={'projection':'polar'})
ax2.set_theta_zero_location("N")
ax2.set_ylim(-0,40)
ax2.set_theta_direction(-1)
ax2.scatter(d4clus['azimut'], np.sqrt(d4clus['vlon']**2+d4clus['vlat']**2),
            color=colors[y_pred])
ax2.set_title('Azimut and V horizontal', fontsize=16)
fig2.savefig(f'output/strain_rot/{period}_polar.png', dpi=360,
             bbox_inches='tight')

# figure 3
fig3, ax3 = plt.subplots(1, 1, figsize=(15,5))
ax3.scatter(d4clus['vlon'], d4clus['vlat'], s=30, color=colors[y_pred])
ax3.set_xlabel('vlon')
ax3.set_ylabel('vlat')
fig3.savefig(f'output/strain_rot/{period}_xy.png', dpi=360, bbox_inches='tight')

# figure 4
colMap = dict(enumerate(colors))
cols = list(map(lambda x:colMap.get(x), y_pred))
clusters_df2 = d4clus[['S_inv1', 'S_inv3', 'wz']]
fig42 = plt.figure(42, figsize=(15,15))
ax42 = plt.gca()
scatter_matrix(clusters_df2, c=cols, diagonal='kde', ax=ax42, alpha=0.8)
fig42.savefig(f'output/strain_rot/{period}_matrix.png', dpi=360,
              bbox_inches='tight')

# figure 5
scale=0.1
fmt = r'%2.0f$\degree$' # Format you want the ticks, e.g. '40%'
# pintar la tierra con un color plano
land = cfeature.NaturalEarthFeature(category='physical',
                            name='land', scale='10m',
                            edgecolor='face',
                            facecolor=cfeature.COLORS['land'])

fig6, ax6 = plt.subplots(1, 1, figsize=(15,24),
                         subplot_kw={'projection': ccrs.PlateCarree()})

# cycle through the clusters
labels = np.unique(y_pred)
#ax6.scatter(d4clus['lon'], d4clus['lat'], s=60, color=colors[y_pred])
ax6.scatter(d4clus['lon'], d4clus['lat'], s=60, c=y_pred, cmap = 'tab20')

# especificar ubicación de los ticks
ax6.set_xticks(meridianos)
ax6.set_yticks(paralelos)
ax6.set_extent(extent)
ax6.tick_params(labelsize=13)

ticks = mticker.FormatStrFormatter(fmt)
ax6.xaxis.set_major_formatter(ticks)
ax6.yaxis.set_major_formatter(ticks)

ax6.add_feature(land, edgecolor='lightgray', alpha=1, zorder=0)

# agregar lineas de costa
ax6.coastlines(resolution='10m')

scale_bar(ax6, 500, location=(0.8, 0.93))
fig6.canvas.draw()
fig6.savefig(f'output/strain_rot/{period}_map_geo_{n_clusters}.png', dpi=200,
             bbox_inches='tight')
