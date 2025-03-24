# -*- coding: utf-8 -*-
"""
Funciones para calcular el campo de desplazamientos 2D de una viga con las
ecuaciones propuestas por Kartunnen & Von Hertzen.

Por: Alejandro Hincapié Giraldo
"""
import numpy as np
import matplotlib.pyplot as plt

def despl_2d(e, LaG, qq, L, nuu, E, hh, I, idx, a, a_r):
    ''' Permite calcular el campo de desplazamientos 2D de un elemento finito
        de viga, a partir de sus desplazamientos nodales.
    '''
    Le = L[e]       # longitud del elemento finito e
    qe = -qq[e, 0]   # Carga distribuida
    nu = nuu[e]     # Coeficiente de Poisson 
    Ee = E[e]       # Módulo de elasticidad
    h = hh[e]       # Altura viga
    Ie = I[e]       # Inercia de la sección
    
    n = 10  # Número de puntos verticales para evaluar desplazamientos
    y = np.linspace(-h/2, h/2, n)

    x1, x2 = [-Le/2, Le/2]
    
    v1, t1, v2, t2 = a[idx[e]]
    u1, u2 = a_r[LaG[e]]
    u = np.array([u1, v1, t1, u2, v2, t2])
    
    Phi = 3*h**2*(1+nu)/Le**2
    
    def Ux(x,y):
        # Campo de desplazamientos horizontales.
        
        Nx = np.array([(Le/2 - x)/Le, y*(3*Le**2 + 4*nu*y**2 - 12*x**2 + 8*y**2)/(2*Le**3*(9*Phi + 1)), y*(-18*Le**2*Phi + Le**2 + 36*Le*Phi*x + 4*Le*x + 4*nu*y**2 - 12*x**2 + 8*y**2)/(4*Le**2*(9*Phi + 1)), (Le/2 + x)/Le, y*(-3*Le**2 - 4*nu*y**2 + 12*x**2 - 8*y**2)/(2*Le**3*(9*Phi + 1)), -y*(18*Le**2*Phi - Le**2 + 36*Le*Phi*x + 4*Le*x - 4*nu*y**2 + 12*x**2 - 8*y**2)/(4*Le**2*(9*Phi + 1))])
        L1 = x*y*(-Le**2 - 4*nu*y**2 + 4*x**2 - 8*y**2)/(24*Ee*Ie)
        
        return Nx@u + qe*L1
    
    def Uy(x,y):
        # Campo de desplazamientos verticales.
        
        Ny = np.array([nu*y/Le, (9*Le**3*Phi + Le**3 - 18*Le**2*Phi*x - 3*Le**2*x + 12*nu*x*y**2 + 4*x**3)/(2*Le**3*(9*Phi + 1)), (9*Le**3*Phi + Le**3 - 2*Le**2*x - 36*Le*Phi*nu*y**2 - 36*Le*Phi*x**2 - 4*Le*nu*y**2 - 4*Le*x**2 + 24*nu*x*y**2 + 8*x**3)/(8*Le**2*(9*Phi + 1)), -nu*y/Le, (9*Le**3*Phi + Le**3 + 18*Le**2*Phi*x + 3*Le**2*x - 12*nu*x*y**2 - 4*x**3)/(2*Le**3*(9*Phi + 1)), (-9*Le**3*Phi - Le**3 - 2*Le**2*x + 36*Le*Phi*nu*y**2 + 36*Le*Phi*x**2 + 4*Le*nu*y**2 + 4*Le*x**2 + 24*nu*x*y**2 + 8*x**3)/(8*Le**2*(9*Phi + 1))])
        L2 = (-Le**4 - 12*Le**2*h**2*nu - 12*Le**2*h**2 + 8*Le**2*nu*y**2 + 8*Le**2*x**2 + 16*h**3*nu**2*y - 16*h**3*y + 24*h**2*nu**2*y**2 + 48*h**2*nu*x**2 + 48*h**2*x**2 - 24*h**2*y**2 - 96*nu*x**2*y**2 + 32*nu*y**4 - 16*x**4 + 16*y**4)/(384*Ee*Ie)
        
        return Ny@u + qe*L2
    
    U1 = np.zeros((n, 2))  # Desplazamientos en x = x1
    U2 = np.zeros((n, 2))  # Desplazamientos en x = x2
    
    for i in range(n):
        Ux1_i = Ux(x1, y[i])
        Ux2_i = Ux(x2, y[i])
        Uy1_i = Uy(x1, y[i])
        Uy2_i = Uy(x2, y[i])
        
        U1[i] = [Ux1_i, Uy1_i]
        U2[i] = [Ux2_i, Uy2_i]
    
    return y, U1, U2

def coord_deformadas(e, LaG, qq, L, nuu, E, hh, I, idx, a, a_r, xnod, escala):
    ''' Permite obtener las coordenadas desplazadas de los nodos de un elemento
        finito bidimensional de viga, a partir de las ecuaciones deducidas de
        (Kartunnen & Von Hertzen).
    '''
    y, U1, U2 = despl_2d(e, LaG, qq, L, nuu, E, hh, I, idx, a, a_r)
    
    n = y.size
    
    x1, x2 = xnod[LaG[e]]
    
    # Se calculan las nuevas posiciones de los nodos:
    
    coord_1 = np.c_[np.tile(x1, n), y]
    coord_2 = np.c_[np.tile(x2, n), y]
    
    ncoord_1 = coord_1 + U1*escala
    ncoord_2 = coord_2 + U2*escala
    
    return ncoord_1, ncoord_2


def plot_despl(e, LaG, xnod, qq, L, nuu, E, hh, I, idx, a, a_r, ax, escala):
    ''' Grafica los desplazamientos de un elemento finito bidimensional de viga.
    '''
    y, U1, U2 = despl_2d(e, LaG, qq, L, nuu, E, hh, I, idx, a, a_r)
    n = y.size
    x1, x2 = xnod[LaG[e]]
    ncoord_1, ncoord_2 = coord_deformadas(e, LaG, qq, L, nuu, E, hh, I, idx,
                                          a, a_r, xnod, escala)
    
    for i in range(n):
        ax.plot([x1, x2], [y[i], y[i]], 'ro-', lw=0.5, markersize=1)
        ax.plot([ncoord_1[i, 0], ncoord_2[i, 0]], 
                [ncoord_1[i, 1], ncoord_2[i, 1]], 'b.-', lw=1)