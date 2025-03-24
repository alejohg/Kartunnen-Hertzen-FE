#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programa para resolver una viga de Euler-Bernoulli por el método de los
elementos finitos.

Creado por: Alejandro Hincapié G
"""

import pandas as pd
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline 

# %% defino las constantes y variables
Y = 0
TH = 1

filename = 'viga'

# %% Defino las funciones:

def extrapolar_mom(e):
    ''' Extrapola de manera lineal los momentos flectores de los puntos de
        Gauss del EF número e a los nodos correspondientes a tal EF.
    '''
    x = xnod[LaG[e]]  # Coordenadas de los nodos del EF
    M = mom[:, e]     # Momentos en los puntos de Gauss
    xg = xmom[:, e]   # Coordenadas de los puntos de Gauss

    f = InterpolatedUnivariateSpline(xg, M, k=1)  # Función de extrap. lineal
    return f(x)


def plot_momento(e, ax):
    ''' Grafica el momento flector para el elemento finito "e". La gráfica se
        realiza sobre el Axes especificado como "ax".
    '''
    x = xnod[LaG[e]]  # Coordenadas de los nodos del EF
    M = extrapolar_mom(e)  # Momentos en los nodos del EF

    transicion = np.sum(M >= 0) == 1  # Se determina si el momento cambia
                                      # de signo dentro del EF

    if transicion:
        # Si el momento cambia de signo en el EF, se deben graficar el tramo
        # positivo y el negativo por aparte:
        m = ((np.diff(M))/(np.diff(x)))[0]  # Pendiente de la recta
        x_0 = x[0] - M[0]/m               # Punto donde el momento es 0

        # Se definen los dos tramos a graficar:
        t_1 = [x[0], x_0]  # Primer tramo
        t_2 = [x_0, x[1]]  # Segundo tramo

        # Se grafica el tramo 1:
        ax.plot(t_1, [M[0], 0], lw=1,
                color='red' if M[0] <= 0 else 'green')
        ax.fill_between(t_1, [M[0], 0],
                        color='salmon' if M[0] <= 0 else 'lightgreen')

        # Se grafica el tramo 2:
        ax.plot(t_2, [0, M[1]], lw=1,
                color='red' if M[1] <= 0 else 'green')
        ax.fill_between(t_2, [0, M[1]],
                        color='salmon' if M[1] <= 0 else 'lightgreen')

    else:
        ax.plot(x, extrapolar_mom(e), lw=1,
                 color='red' if np.all(mom[:,e] <= 0) else 'green')
        ax.fill_between(x, extrapolar_mom(e),
                     color='salmon' if np.all(mom[:,e] <= 0) else 'lightgreen')


# %% Se lee el archivo de Excel
df       = pd.read_excel(f'{filename}.xlsx', sheet_name=None)


# %% se lee la posicion de los nodos
T = df['xnod'][['nodo', 'x']].set_index('nodo')
xnod    = T['x'].sort_index().to_numpy() # posicion de los nodos
L       = np.diff(xnod)                  # longitud de cada EF

nno  = xnod.size                   # numero de nodos
nef  = nno - 1                     # numero de elementos finitos (EF)
ngdl = 2*nno                       # numero de grados de libertad
gdl  = np.reshape(np.arange(ngdl), (nno, 2))  # grados de libertad
 

# %% se leen la matriz de conectividad (LaG), el modulo de elasticidad, las
# propiedades del material y las cargas
T     = (df['LaG_EI_q'][['EF', 'NL1', 'NL2', 'E', 'I', 'q1e', 'q2e']]
         .fillna(0).set_index('EF').sort_index())

LaG   = T[['NL1','NL2']].to_numpy() - 1   # Relación de cada EF con sus nodos
E     = T['E'].to_numpy()                 # modulo de elasticidad E del EF
I     = T['I'].to_numpy()                 # momento de inercia Iz del EF
qq     = T[['q1e', 'q2e']].to_numpy()      # relación de las cargas distribuidas


# %% relacion de los apoyos
T       = (df['restric'][['nodo', 'direccion', 'desplazamiento']]
           .set_index('nodo').sort_index())
dirdesp = T['direccion'].to_numpy() - 1
ac      = T['desplazamiento'].to_numpy()  # desplazamientos conocidos


# %% grados de libertad del desplazamiento conocidos y desconocidos
n_apoyos = T.shape[0]
c = np.zeros(n_apoyos, dtype=int)  # GDL conocidos

for i in range(n_apoyos):
    c[i] = gdl[T.index[i]-1][dirdesp[i]]
d = np.setdiff1d(np.arange(ngdl), c)       # GDL desconocidos


# %% relación de cargas puntuales
T = (df['carga_punt'][['nodo', 'direccion', 'fuerza_puntual']]
     .set_index('nodo').sort_index())
nfp   = T.shape[0]
dirfp = T['direccion'].to_numpy() - 1
fp    = T['fuerza_puntual'].to_numpy()



# %% se colocan las fuerzas/momentos nodales en el vector de fuerzas nodales
#  equivalentes global "f"
f = np.zeros(ngdl)   # vector de fuerzas nodales equivalentes global
for i in range(nfp):
    f[gdl[T.index[i]-1][dirfp[i]]] = fp[i]


# %% relacion de los resortes

T       = df['resortes'][['nodo', 'tipo', 'k']].set_index('nodo').sort_index()
tipores = T['tipo'].to_numpy() - 1  # Y=1 (vertical), TH=2 (rotacional)
kres    = T['k'].to_numpy()         # constante del resorte

# %% grados de libertad del desplazamiento conocidos y desconocidos
K = np.zeros((ngdl, ngdl))  # matriz de rigidez global
n_resortes = len(kres)

for i in range(n_resortes):
    idx = gdl[T.index[i]-1][tipores[i]]
    K[idx, idx] += kres[i]


# %% VIGA DE EULER-BERNOULLI:
# Con el programa "func_forma_euler_bernoulli.m" se calcularon:
#   Ke     = la matriz de rigidez de flexion del elemento e
#   fe     = el vector de fuerzas nodales equivalentes
#   Bb     = la matriz de deformaciones de flexion
#   N      = matriz de funciones de forma
#   dN_dxi = derivada de la matriz de funciones de forma con respecto a xi

# %% ensamblo la matriz de rigidez global y el vector de fuerzas nodales
#  equivalentes global para la viga de Euler-Bernoulli

idx = [None]*nef      # grados de libertad del elemento e

for e in range(nef):  # ciclo sobre todos los elementos finitos
    idx[e] = np.r_[gdl[LaG[e, 0]], gdl[LaG[e, 1]]]
    Le = L[e]
    w1, w2 = qq[e]
    # Matriz de rigidez de flexion del elemento e y...
    # vector de fuerzas nodales equivalentes de una carga trapezoidal:
    

    Ke = E[e]*I[e]/Le**3 * np.array([[12,   6*Le,   -12,   6*Le],
                                     [6*Le, 4*Le**2, -6*Le, 2*Le**2],
                                     [-12,  -6*Le,    12,  -6*Le],
                                     [6*Le, 2*Le**2, -6*Le, 4*Le**2]])
    
    fe = np.array([(Le*(7*w1 + 3*w2))/20,       # = Y1
                   (Le**2*(3*w1 + 2*w2))/60,    # = M1
                   (Le*(3*w1 + 7*w2))/20,       # = Y2
                  -(Le**2*(2*w1 + 3*w2))/60])   # = M2


    # se ensambla la matriz de rigidez K y el vector de fuerzas nodales
    # equivalentes f
    K[np.ix_(idx[e], idx[e])] += Ke
    f[idx[e]] += fe


# %% se resuelve el sistema de ecuaciones
# f = vector de fuerzas nodales equivalentes
# q = vector de fuerzas nodales de equilibrio del elemento
# a = desplazamientos

# |   qd   |   | Kcc Kcd || ac |   | fd |
# |        | = |         ||    | - |    |
# | qc = 0 |   | Kdc Kdd || ad |   | fc |

# %% extraigo las submatrices y especifico las cantidades conocidas
Kcc = K[np.ix_(c, c)];  Kcd = K[np.ix_(c, d)]; fd = f[c]
Kdc = K[np.ix_(d, c)];  Kdd = K[np.ix_(d, d)]; fc = f[d]

ad = np.linalg.solve(Kdd, fc - Kdc@ac)   # calculo desplazamientos desconocidos
qd = Kcc@ac + Kcd@ad - fd                # calculo fuerzas de equilibrio desc.
a = np.zeros(ngdl);  a[c] = ac;  a[d] = ad  # desplazamientos 
q = np.zeros(ngdl);  q[c] = qd              # fuerzas nodales equivalentes


# %% calculo de los momentos flectores
# (se calcula el momento en el centro de cada elemento finito)
# se reserva la memoria
# recuerde que en cada elemento se calculan los momentos en las raices de
# los polinomios de Legendre de grado dos

xmom = np.empty((2, nef))  # posicion donde se calcula
mom  = np.empty((2, nef))  # momento flector
cor  = np.empty(nef)       # fuerza cortante
xi = np.array([-sqrt(1/3), sqrt(1/3)])  # raices del P. de Legendre de grado 2


#%%
for e in range(nef):

    Le = L[e]  # longitud del elemento finito e

    # matriz de deformaciones de flexion
    
    #----------------IMPORTANTE PROBLEMA AQUÍ----------------------------------
    Bbe = np.array([(6*xi)/Le**2, (3*xi - 1)/Le, -(6*xi)/Le**2, (3*xi + 1)/Le])

    
    # lugar donde se calcula el momento (centro del EF)
    xmom[:, e] = Le*xi/2 + (xnod[LaG[e, 0]] + xnod[LaG[e, 1]])/2

    # vector de desplazamientos nodales del elemento a^{(e)}
    ae = a[idx[e]]

    mom[:, e] = E[e]*I[e]*Bbe.T@ae  # momento flector
    dN3_dxi3  = np.array([3/2, (3*Le)/4, -3/2, (3*Le)/4])
    cor[e]    = E[e]*I[e]*(8/(Le**3))*dN3_dxi3.T@ae  # fuerza cortante


# %% se calculan los desplazamientos al interior de cada EF
nint = 10                      # No. de puntos donde se interpolará en el EF
xi = np.linspace(-1, 1, nint)  # coordenadas naturales

xx = nef*[None]  # interpol de posiciones (geometria) en el elemento
ww = nef*[None]  # interpol desplazamientos en el elemento
tt = nef*[None]  # interpol angulo en el elemento

for e in range(nef):  # ciclo sobre todas los elementos finitos

    Le = L[e]  # longitud del elemento finito e

    # Matriz de funciones de forma y su derivada:
    N = np.array([xi**3/4 - (3*xi)/4 + 1/2,
                  -(Le*(- xi**3/4 + xi**2/4 + xi/4 - 1/4))/2,
                  - xi**3/4 + (3*xi)/4 + 1/2,
                  -(Le*(- xi**3/4 - xi**2/4 + xi/4 + 1/4))/2])

    dN_dxi = np.array([(3*xi**2)/4 - 3/4,
                       -(Le*(- (3*xi**2)/4 + xi/2 + 1/4))/2,
                       3/4 - (3*xi**2)/4,
                       (Le*((3*xi**2)/4 + xi/2 - 1/4))/2])

    # vector de desplazamientos nodales del elemento a^{(e)}
    ae = a[idx[e]]

    # interpola sobre la geometria (coord naturales a geometricas)
    xx[e] = Le*xi/2 + (xnod[LaG[e, 0]] + xnod[LaG[e, 1]])/2

    # se calcula el desplazamiento al interior del elemento finito
    ww[e] = N.T@ae

    # se calcula el angulo al interior del elemento finito
    tt[e] = np.arctan((dN_dxi.T*2/Le)@ae)


# %% imprimo los resultados:

print('Desplazamientos nodales')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

vect_mov = np.reshape(a, (nno, 2))  # vector de movimientos
for i in range(nno):
    print(f'Nodo {i+1}: w = {vect_mov[i,Y]*1000:.3g} mm, '
          f'theta = {vect_mov[i,TH]:3g} rad \n')

print()
print('Fuerzas nodales de equilibrio (solo se imprimen en los apoyos)')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
q = np.reshape(q, (nno, 2))
for i in range(nno):
    if not np.all(q[i] == 0):
        print(f'Nodo {i+1}: Ry = {q[i, Y]:3f} kN, Mz = {q[i, TH]:3f} kN-m')


# %% Finalmente se grafican los resultados

fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 12))

fig1.tight_layout(pad=7.0)
fig2.tight_layout(pad=7.0)


ax1.set_title('Solucion con el MEF para el desplazamiento')
ax1.set_xlabel('Eje X (m)')                # titulo del eje X
ax1.set_ylabel('Desplazamiento (mm)')       # titulo del eje Y
ax1.set_xlim(xnod[0], xnod[-1])            # rango en el eje X del grafico)

ax2.set_title('Solucion con el MEF para el giro')
ax2.set_xlabel('Eje X (m)')                # titulo del eje X
ax2.set_ylabel('Giro (rad)')       # titulo del eje Y
ax2.set_xlim(xnod[0], xnod[-1])            # rango en el eje X del grafico)

ax3.set_title('Solucion con el MEF para el momento flector')
ax3.set_xlabel('Eje X (m)')                # titulo del eje X
ax3.set_ylabel('Momento flector (kN-m)')   # titulo del eje Y
ax3.set_xlim(xnod[0], xnod[-1])            # rango en el eje X del grafico)

ax4.set_title('Solucion con el MEF para la fuerza cortante')
ax4.set_xlabel('Eje X (m)')                # titulo del eje X
ax4.set_ylabel('Fuerza cortante (kN)')     # titulo del eje Y
ax4.set_xlim(xnod[0], xnod[-1])            # rango en el eje X del grafico)

# Se imprimen los respectivos diagramas:
for ax in [ax1, ax2, ax3, ax4]:
    ax.axhline(0, lw=0.7, color='k')


for e in range(nef):
    x = xnod[LaG[e]]  # Coordenadas de los nodos del EF
    ax1.plot(xx[e], ww[e]*1000, 'b-', lw=1)  # Desplazamientos verticales
    ax2.plot(xx[e], tt[e], 'b-', lw=1)       # Ángulo de giro

    plot_momento(e, ax3)  # Se grafica el momento

    # Fuerza cortante:
    ax4.plot(x, [cor[e], cor[e]], lw=1, color='red' if cor[e]<=0 else 'green')
    ax4.fill_between(x, [cor[e], cor[e]],
                     color='salmon' if cor[e]<=0 else 'lightgreen')

# %% Se exportan los resultados:

# Se crea el vector de momentos extrapolados a los nodos:
mom_ex = np.empty((nef, 2))
for e in range(nef):
    mom_ex[e] = extrapolar_mom(e)

despl = a.reshape(nno,2)

np.savetxt('./resultados_EB/despl_EB.csv', despl)
np.savetxt('./resultados_EB/M_EB.csv', mom_ex)
np.savetxt('./resultados_EB/Q_EB.csv', cor)
np.savetxt('xnod.csv', xnod)

