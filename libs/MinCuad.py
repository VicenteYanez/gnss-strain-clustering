__doc__ = """
GF5013 - Metodos Inversos Aplicados a la Geofisica
Primavera 2017
Prof. Francisco Hernán Ortega Culaciati
ortega.francisco@u.uchile.cl
Departamento de Geofísica - FCFM
Universidad de Chile

02 de Octubre de 2017

"""
from numpy.linalg import lstsq 
import numpy as np


######
def MinCuadSimple(G, d):
    """
    Calcula la solución de G*m = d por el método de mínimos cuadrados simples. 
    Devuelve un diccionario con m y Cm. 
    Para el cálculo de la matriz de covarianza del modelo, se asume que los 
    errores del ajuste (i.e., de la diferencia d-Gm) son i.i.d. con media nula 
    y varianza unitaria.  
    """
    Ndata, Npar = G.shape
    Cm = np.linalg.lstsq( G.T.dot(G), np.eye(Npar), rcond = None )[0]
    m = Cm.dot( G.T.dot(d) )
    return {'m' : m, 'Cm': Cm}


######
def MinCuadPesos(G, d, Wx):
    """
    Calcula la solución de Wx*G*m = Wx*d por el método de mínimos cuadrados (con pesos). 
    Devuelve un diccionario con m y Cm. 
    Para el cálculo de la matriz de covarianza de los parámetros estimados del modelo, 
    se asume que los errores del ajuste (i.e., de la diferencia d-Gm) siguen una 
    distribución normal multivariada con media nula y matriz de covarianza Cx.
    donde Cx = inv( Wx.T.dot(Wx) ) 
    """
    return MinCuadSimple( Wx.dot(G), Wx.dot(d) )

