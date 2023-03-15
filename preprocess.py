
import sys, os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DOMAIN_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pickle

from libs.field import evidence_velocity_gradients, invariants_and_wz


if __name__ == '__main__':

    use_multiprocessing = True
    factor_velo = 1E-6 # as velocities in datafiles are in mm/year
    num_proc = None

    # alpha range to test
    alpha_range = np.arange(10,650, 2.5)
    print(len(alpha_range), alpha_range)

    data = 'post'
    # dataset 2 load
    if data == 'pre':
        datafile = './data/pre2014.txt'
        gps_data = pd.read_csv(datafile, sep='  ',
                            names=['lon', 'lat', 'vlon', 'vlat', 'elon', 'elat'])
    elif data == 'post':
        datafile = './data/velocidades_sam.txt'
        gps_data = pd.read_csv(datafile, sep=' ', names=['station', 'lon', 'lat',
                                                        'vlon', 'vlat', 'vup',
                                                        'elon', 'elat', 'eup',
                                                        'start', 'end', 'tspan'])
        gps_data.drop(['vup', 'eup', 'start', 'end', 'tspan'], axis=1, inplace=True)


    gps_data['station'] = np.arange(0, gps_data.shape[0])
    gps_data.set_index('station', inplace=True)
    gps_data['azimut'] = np.arctan2(gps_data['vlon'], gps_data['vlat'])

    # scale the data
    gps_data['h'] = np.sqrt(gps_data['vlon']**2 + gps_data['vlat']**2)
    gps_data['x_unit'] = gps_data['vlon']/gps_data['h']
    gps_data['y_unit'] =gps_data['vlat']/gps_data['h']

    X = np.array(gps_data[['vlon', 'vlat']])
    X_std = np.array(gps_data[['elon', 'elat']])

    m, evidence, alpha_sta, evi_sta, valid_sta = evidence_velocity_gradients(
        X, X_std, gps_data['lon'], gps_data['lat'], alpha_range,
        factor_velo = factor_velo, 
            use_multiprocessing = use_multiprocessing, 
            num_proc = num_proc)
    features = invariants_and_wz(m)

    print('features keys = ', features.keys())

    # save tensor params in the dataframe
    gps_data['t1'] = m[:,0,0]
    gps_data['t2'] = m[:,1,0]
    gps_data['vxdx'] = m[:,2,0]
    gps_data['vxdy'] = m[:,3,0]
    gps_data['vydx'] = m[:,4,0]
    gps_data['vydy'] = m[:,5,0]
    gps_data['S_inv1'] = features['inv1']
    gps_data['S_inv3'] = features['inv3']
    gps_data['wz'] = features['wz']
    gps_data['ev11'] = features['evectors'][:,0,0]
    gps_data['ev12'] = features['evectors'][:,0,1]
    gps_data['ev21'] = features['evectors'][:,1,0]
    gps_data['ev22'] = features['evectors'][:,1,1]
    gps_data['evalue1'] = features['evalues'][:,0]
    gps_data['evalue2'] = features['evalues'][:,1]
    gps_data['s_max'] = features['s_max']
    gps_data['valid_sta'] = np.asarray(valid_sta, dtype = int)
    print(gps_data)
    gps_data.to_csv(f'data/ready4clustering_{data}.csv')

    tosave = {}
    tosave['alpha_range'] = alpha_range
    tosave['data'] = data
    tosave['datafile'] = datafile
    tosave['gps_data'] = gps_data
    tosave['X'] = X
    tosave['X_std'] = X_std
    tosave['m'] = m
    tosave['evidence'] = evidence
    tosave['alpha_sta'] = alpha_sta
    tosave['evi_sta'] = evi_sta
    tosave['features'] = features
    tosave['valid_sta'] = valid_sta

    with open(f'data/ready4clustering_{data}.pickle', 'wb') as fileOUT:
        pickle.dump(tosave, fileOUT, protocol = 5)

    # ##############################################################################
    # Evidence Plots
    # ##############################################################################

    # fig 1.1
    fig1 = plt.figure(figsize=(10,10))
    ax1 = plt.gca()
    ax1.plot(alpha_range, evidence)
    for s, i_evi in enumerate(evidence.T):
        i_max = np.nanargmax(i_evi)
        ax1.scatter(alpha_range[i_max], evidence[i_max, s], s=20, marker='s',
                    zorder=10)
    fig1.savefig(f'output/strain_rot/evidence_{data}.png', dpi=360,
                bbox_inches='tight')

    # fig 2
    fig2 = plt.figure(figsize=(10,10))
    ax2 = plt.gca()
    i_max = np.nanargmax(evidence, axis=0)[np.newaxis:,]
    ratios = evidence #- evidence[i_max]
    ax2.plot(alpha_range, ratios)
    fig2.savefig(f'output/strain_rot/evidence_ratio_{data}.png', dpi=360,
                bbox_inches='tight')

    # fig 3: evidence map
    fig3 = plt.figure(figsize=(10,10))
    ax3 = fig3.add_subplot(projection=ccrs.PlateCarree())
    ax3_cbar = fig3.add_axes([0.7, 0.52, .0075, .08], facecolor='w', zorder=2)


    land = cfeature.NaturalEarthFeature(category='physical',
                                name='land', scale='10m',
                                edgecolor='face',
                                facecolor=cfeature.COLORS['land'])

    extent = (-77, -57, -48, -17)  # map extent
    paralelos = np.arange(-48, -17, 2)
    meridianos = np.arange(-77, -57, 2)
    fmt = r'%2.0f$\degree$' # Format you want the ticks, e.g. '40%'
    ticks = mticker.FormatStrFormatter(fmt)

    ax3.set_xticks(meridianos)
    ax3.set_yticks(paralelos)
    ax3.set_extent(extent)
    ax3.tick_params(labelsize=10)
    ax3.xaxis.set_major_formatter(ticks)
    ax3.yaxis.set_major_formatter(ticks)

    ax3.add_feature(land, edgecolor='lightgray', alpha=1, zorder=0)

    sc3 = ax3.scatter(gps_data['lon'], gps_data['lat'], s=25,
                    c=evi_sta, cmap='nipy_spectral',
                    edgecolors='k', linewidths=0.3, alpha=1.)
    cb3 = plt.colorbar(sc3, cax=ax3_cbar, fraction=0.046, pad=0.04,
                    label='evidence')

    #fig3.savefig(f'output/strain_rot/evidence_map.png', dpi=360,
    #             bbox_inches='tight')

    # fig 4: alpha map
    fig4 = plt.figure(figsize=(10,10))
    ax4 = fig4.add_subplot(projection=ccrs.PlateCarree())
    ax4_cbar = fig4.add_axes([0.7, 0.52, .0075, .08], facecolor='w', zorder=2)

    ax4.set_xticks(meridianos)
    ax4.set_yticks(paralelos)
    ax4.set_extent(extent)
    ax4.tick_params(labelsize=10)
    ax4.xaxis.set_major_formatter(ticks)
    ax4.yaxis.set_major_formatter(ticks)

    ax4.add_feature(land, edgecolor='lightgray', alpha=1, zorder=0)

    sc4 = ax4.scatter(gps_data['lon'], gps_data['lat'], s=25,
                    c=alpha_sta, cmap='nipy_spectral',
                    edgecolors='k', linewidths=0.3, alpha=1.)
    cb4 = plt.colorbar(sc4, 
                    label='evidence')

    fig4.savefig(f'output/strain_rot/alpha_map_{data}.png', dpi=360,
                bbox_inches='tight')


    # fig 5: features map
    features2plot = ['S_inv1', 'S_inv3', 'wz', 's_max', 'valid_sta', 't1', 't2']
    for feature in features2plot:
        fig5 = plt.figure(figsize=(10,10))
        ax5 = fig5.add_subplot(projection=ccrs.PlateCarree())
        ax5_cbar = fig5.add_axes([0.7, 0.52, .0075, .08], facecolor='w', zorder=2)

        ax5.set_xticks(meridianos)
        ax5.set_yticks(paralelos)
        ax5.set_extent(extent)
        ax5.tick_params(labelsize=10)
        ax5.xaxis.set_major_formatter(ticks)
        ax5.yaxis.set_major_formatter(ticks)

        ax5.add_feature(land, edgecolor='lightgray', alpha=1, zorder=0)

        sc5 = ax5.scatter(gps_data['lon'], gps_data['lat'], s=25,
                        c=gps_data[feature], cmap='nipy_spectral',
                        edgecolors='k', linewidths=0.3, alpha=1.)
        cb5 = plt.colorbar(sc4, cax=ax4_cbar, fraction=0.046, pad=0.04,
                        label='evidence')
        fig5.savefig(f'output/strain_rot/{feature}_map_{data}.png', dpi=360,
                    bbox_inches='tight')
        plt.close()
