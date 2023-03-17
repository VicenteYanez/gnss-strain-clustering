"""
Francisco Hernán Ortega Culaciati
ortega.francisco@uchile.cl
Departamento de Geofísica - FCFM
Universidad de Chile

02 de Octubre de 2017

"""
from numpy.linalg import lstsq 
import numpy as np


######
def MinCuadSimple(G, d):
    """
    Estimates the unknown parameter vector m in G*m = d + eta using simple linear least
    squares. Here eta is the error of the misfit (including data and model prediction
    errors), assumed to be i.i.d. with zero mean and a variance with unitary value.
    Returns a dictionary with keys 'm' and 'Cm' whose values are the estimated model m and
    its a posteriori covariance matrix Cm, respectively.
    The solution of this problem is a Maximum Likelihood solution if misfit errors are
    i.i.d. with zero mean and a variance with unitary value. 
    """
    Ndata, Npar = G.shape
    Cm = np.linalg.lstsq( G.T.dot(G), np.eye(Npar), rcond = None )[0]
    m = Cm.dot( G.T.dot(d) )
    return {'m' : m, 'Cm': Cm}


######
def MinCuadPesos(G, d, Wx):
    """
    Estimates the unknown parameter vector m in Wx*G*m = Wx*d + eta using simple linear
    least squares. eta is the error of the misfit (including data and model prediction
    errors), assumed to be i.i.d. with zero mean and a variance with unitary value.
    Returns a dictionary with keys 'm' and 'Cm' whose values are the estimated model m and
    its a posteriori covariance matrix Cm, respectively.
    The solution of this problem is a Maximum Likelihood solution if Wx is such that
    Wx.T.dot(Wx) = inv(Cx) where Cx = Cd + Cp is the covariance matrix of the misfit, and
    Cd, Cp are the covariance matrices of the data and model prediction, respectively.
    """
    return MinCuadSimple( Wx.dot(G), Wx.dot(d) )

