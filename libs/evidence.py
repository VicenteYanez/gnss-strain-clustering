
import numpy as np
from scipy import stats

from .MinCuad import MinCuadPesos


def evidence_wleastsquares(G, d_obs, W):
    """
    Calculates the evidence for a estimation of weighted least squares

    Parameters
    ----------
    G : (Nd, M) numpy array
        Kernel of shape Nd (number of observations) and M (number of parameters)
    d_obs : (Nd, 1) numpy array
        Observational data
    Wd : (Nd, Nd) numpy array
        Weight matrix of the data

    Returns
    -------
    log_evidence : float
        Calculated log of the evidence
    """
    #print('G = ', G)
    #print('d_obs = ', d_obs)
    #print('W = ', W)
    # least square
    m_Cm = MinCuadPesos(G, d_obs, W)
    m = m_Cm['m']
    Cm = m_Cm['Cm']
    #print('m = ', m)
    #print('Cm = ', Cm)

    Nd, M = G.shape[0], G.shape[1]

    # RTR
    invCd = np.dot(W.T, W)
    R = np.dot(G,m) - d_obs
    RTR = float(np.dot(R.T, invCd).dot(R))  # 1d array can be converted to float
    #print('RTR = ', RTR)
    # det Cd, Cm
    log_det_Cd = -np.linalg.slogdet(invCd)[1]  # log |Cd^-1| = -log |Cd|
    #print('log_det_Cd = ', log_det_Cd)
    log_det_Cm = np.linalg.slogdet(Cm)[1]
    #print('log_det_Cm = ', log_det_Cm)

    # evidence
    log_evidence = -0.5*RTR + 0.5*(log_det_Cm - (Nd-M)*np.log(2*np.pi))
    log_evidence -= 0.5*log_det_Cd
    #print('log_evidence = ', log_evidence)
    return m, Cm, log_evidence


