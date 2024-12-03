import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear_model(x,m,c):
    return m*x + c

def linear_fit(fitx,fity,a,b):
    """"Fit lineare del set di dati [fitx,fity] nell'intervallo a,b"""
    popt, pcov = curve_fit(linear_model, fitx, fity)
    slope, intercept = popt
    slope_err, intercept_err = np.sqrt(np.diag(pcov))

    X = np.linspace(a,b,100)
    Y = linear_model(X,slope,intercept)
    return X,Y,popt,pcov,slope,intercept,slope_err,intercept_err

def fit_curve(fitx,fity,a,b,f):
    """"Fit curvilineo del set di dati [fitx,fity] tramite la funzione
    f nell'intervallo [a,b]"""
    popt, pcov = curve_fit(f, fitx, fity)
    slope, intercept = popt
    slope_err, intercept_err = np.sqrt(np.diag(pcov))

    X = np.linspace(a,b,100)
    Y = linear_model(X,slope,intercept)
    return X,Y,popt,pcov,slope,intercept,slope_err,intercept_err

def find_roots(X1,Y1,X2,Y2,a,b,order=6,precision=0.001):
    """Trova numericamente le intersezioni dei due set di dati [X1,Y1]
    e [X2,Y2] nell'intervallo [a,b] tramite un fit polinomiale all'orinde order
    (6 di default) con precisione precisione (0.001 di default)"""
    valori = []
    differenze = []

    z1 = np.polyfit(X1, Y1,order)
    p1 = np.poly1d(z1)

    Xfit1 = np.linspace(a,b,1000)
    Yfit1 = p1(Xfit1)
    plt.plot(Xfit1,Yfit1)

    z2 = np.polyfit(X2,Y2,order)
    p2 = np.poly1d(z2)

    Xfit2 = np.linspace(a,b,1000)
    Yfit2 = p2(Xfit2)
    plt.plot(Xfit2,Yfit2)
     
    RICERCA = np.linspace(0, 10, 100000)
    for x in RICERCA:
        if np.abs(p1(x) - p2(x)) <= 0.0001:
            valori.append(x)
            differenze.append(np.abs(p1(x) - p2(x)))

    return valori,differenze