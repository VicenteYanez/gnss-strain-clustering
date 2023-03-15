
import pdb

import numpy as np
import pandas as pd
from scipy.linalg import cholesky


from .geometry import geo2proj, vinc_dist
from .evidence import evidence_wleastsquares

from datetime import datetime



def evidence_velocity_gradients(X, X_std, lons, lats, alpha_range,
                                factor_velo=1E-6, use_multiprocessing = False,
                                num_proc = None):
    """
    Function to estimate the velocity gradient tensor and translation based
    on a modification of Cardozo&Allmendinger (2009) and a evidence based model
    selection to fin the alpha parameter.
    
    factor_velo is a factor to multiply the distance unit of the velocity to
    kilometers (e.g., from if velocities are in mm/year, factor_velo = 1E-6 - as
    v in [mm/year] is equal to 1E-6 v in [km/year] )
    """
    # rearranging X and X_std
    dobs = np.reshape([X[:,0], X[:,1]], (2*len(X[:,0]), 1), order='F')
    std_d = np.reshape([X_std[:,0], X_std[:,1]], (2*len(X_std[:,0]), 1),
                       order='F')

    # empty arrays to fill
    m = np.zeros((len(alpha_range),len(lons),6,1))
    evidence = np.zeros((len(alpha_range),len(lons)))
    valid = np.zeros((len(alpha_range),len(lons)))

    MaxDist = np.max(alpha_range)
    if use_multiprocessing: # parallelize as a function of the alphas
        from multiprocessing import Pool, cpu_count
        if num_proc is None:
            # uses all cpu's
            num_proc = cpu_count()
        # prepare the arguments to spread among the processors
        arguments = [[dobs, std_d, lons, lats, lons, lats,  alpha, MaxDist] 
                     for alpha in alpha_range]
        
        # run in parallel
        with Pool(num_proc) as p:
            par_results = p.map(vg_weigthed2d_multiprocessing, arguments,
                                chunksize = 1)
        # extract the relevant results:
        for a, result in enumerate(par_results):
            
            m_alpha, Cm_alpha, evidence_alpha, valid_alpha = result
            m[a] = m_alpha
            evidence[a] = evidence_alpha  # shape len(grid_x)
            valid[a] = valid_alpha
        
    else:
        # cycle over the alpha choosen range, sequentially
        for a, alpha in enumerate(alpha_range):
            print(f'computing rigid motion and strain for alpha = {alpha}')
            print(' ---> started at:', datetime.now())
            m_alpha, Cm_alpha, evidence_alpha, valid_alpha = vg_weigthed2d(
                dobs, std_d, lons, lats, lons, lats,  alpha, MaxDist)
            m[a] = m_alpha
            evidence[a] = evidence_alpha  # shape len(grid_x)
            valid[a] = valid_alpha
    
    m[:,:,2:,0] *= factor_velo

    # choose the best alpha for each station
    evi_max = np.argmax(evidence, axis=0)  # max evi
    m_best = np.zeros((len(lons),6,1))
    evi_sta = np.zeros(len(lons))
    alpha_sta = np.zeros(len(lons))
    valid_sta = np.zeros(len(lons), dtype = bool)

    # iteration over the station and get the
    for g, sta in enumerate(lons):
        # best model for g-station
        m_best[g] = m[evi_max[g],g,:,:]
        alpha_sta[g] = alpha_range[evi_max[g]]
        evi_sta[g] = evidence[evi_max[g],g]
        valid_sta[g] = valid[evi_max[g],g]

    return m_best, evidence, alpha_sta, evi_sta, valid_sta


def vg_weigthed2d_multiprocessing(arguments):
    """
    arguments must be a tuple or list with all the arguments of vg_weighted2d
    """
    alpha = arguments[6]
    start_time = datetime.now()
    print(f'--> computing rigid motion and strain for alpha = {alpha} ---> started at:', 
            start_time)
    results = vg_weigthed2d(*arguments)
    time_span = datetime.now() - start_time
    print(f'    --> FINISHED calculations for alpha = {alpha} ---> time spent:', 
            time_span)
    return results


def vg_weigthed2d(d_obs, std_d, xi, yi, gridx, gridy, alpha, MaxDist):
    """
    Function that calculates a velocity gradient surface using the
    Grid Distance Weighted from Cardozo&Allmendinger(2009). For the
     least-squares solution it uses the numpy.linalg.lstsq function

    Parameters
    ----------
    d_obs  : velocities vector array like [[v_1x], [v_1y], [v_2x], [v_2y]....]
    std_d  : standard deviations of velocity observations (d_obs)
    xi     : longitude station position 1d array-like
    yi     : latitude station position 1d array-like
    gridx  : list with the x-position of each grid point
    gridy  : list with the y-position of each grid point
    alpha  : constant
    MaxDist: float

    Returns
    ----------
    m_total : np.array
    Cm_total : np.array
    evidence_total : np.array
    valid
    """

    # make distance weighted operator
    m_total = np.zeros((len(gridx), 6, 1))
    Cm_total = np.zeros((len(gridx), 6, 6))
    evidence_total = np.zeros((len(gridx)))
    valid = np.zeros((len(gridx)), dtype = bool)
    for i, x in enumerate(gridx):
        #print(f'******** i = {i:d}  ********')
        # build the kernel 
        G = np.zeros((d_obs.shape[0], 6))
        # proj from geographic to meters
        xp, yp = geo2proj(xi, yi, float(gridx[i]), float(gridy[i]))
        xp, yp = xp/1000, yp/1000  # from meters to km
        for k, x in enumerate(xi):
            G[2*k,:] = [1, 0, xp[k], yp[k], 0, 0]
            G[2*k+1,:] = [0, 1, 0, 0, xp[k], yp[k]]
        # calculate distance to other stations
        d = [vinc_dist(gridy[i], gridx[i],
                       yi[i2], xi[i2]) for i2, x2 in enumerate(xi)]
        d = np.array(d).T[0]  # because vinc_dist return distance and azimuths
        d = d/1000  # m to km
        d = np.reshape([d, d], 2*len(d), order='F')  # order array d
        # calculates the weight matrix
        w_xi = [calc_w_xi(std_di, alpha, di) for di, std_di in zip(d, std_d[:,0])]
        w_xi = np.array(w_xi)

        # Try to filter by number of sites to use in the inversion
        Ikeep = d <= MaxDist 

        if np.sum(Ikeep) <= 8:
            Isorted = np.argsort(d)
            Ikeep = Isorted[0:8]
            print(i, '---> FORCING MINIMUM NUMBER = 8!')
            valid[i] = False
        else:
            valid[i] = True
        
        try:
            # this is done in any case of Ikeep, so I must not update valid[i]
            w_xi_red = w_xi[Ikeep]
            w_xi_red = np.diag(w_xi_red) 
    
            m, Cm, log_evidence = evidence_wleastsquares(
                G[Ikeep,:], d_obs[Ikeep,:], w_xi_red)

        except:
            # if the above does not work, I sill try to do the inversion using
            # all available data, but signal it as valid[i] = False regardless
            # of what dataset was used for the inversion (just to see if such
            # inversion turns out to be something useful)
            print('OOOOPS!')
            w_xi = np.diag(w_xi)
            m, Cm, log_evidence = evidence_wleastsquares(G, d_obs, w_xi)
            valid[i] = False

        m_total[i,:,:] = m
        Cm_total[i,:,:] = Cm
        evidence_total[i] = log_evidence

    return m_total, Cm_total, evidence_total, valid


def invariants_and_wz(m):
    """
    Calculates the invariants and eigenvalues/eigenvectors of the deformational
    tensor, and the vertical vorticity from the matrix m. m has dimension of
    [n station, 6, 1]
    """
    # each gradient have dimension [n station]
    uxdx = m[:,2,0]
    uxdy = m[:,3,0]
    uydx = m[:,4,0]
    uydy = m[:,5,0]

    uxdy_plus_uydx = uxdy + uydx
    uxdy_uydx = uxdy - uydx

    # construct the deformation and rotational tensor
    S = np.zeros((m.shape[0],2,2))
    W = np.zeros((m.shape[0],2,2))
    for j in range(m.shape[0]):
        S[j,:,:] = 0.5*np.array(
            [[2*uxdx[j], uxdy_plus_uydx[j]], [uxdy_plus_uydx[j], 2*uydy[j]]])
        W[j,:,:] = 0.5*np.array([[0, uxdy_uydx[j]], [-uxdy_uydx[j], 0]])
    evalues, evectors = np.linalg.eig(S)

    wz = 2*W[:,1,0]
    wz = 180*wz/np.pi

    inv1 = np.trace(S, axis1=1, axis2=2)
    inv3 = np.linalg.det(S)

    #s_max = (S[:,0,0] - S[:,1,1])/2
    s_max = 0.5 * (evalues[:,0] - evalues[:,1])

    features = {'inv1': inv1, 'inv3': inv3, 'wz': wz, 'evalues': evalues,
                'evectors': evectors, 's_max': s_max} 

    return features


def calc_w_xi(sigma_d, alpha, dist, NsigmasAtAlpha = 3, p = 1,
    threshold = 1E-4):
    """
    computes the misfit weight as a function of distance, exponentially decaying 
    as the distance increases. At distance 0 the weight is 1/sigma_d and decreases
    as the distance increases.
    """
    w_d = (1.0/sigma_d) * np.exp(-((dist/alpha)**p))
    if w_d < threshold:
        w_d = threshold

    return w_d

