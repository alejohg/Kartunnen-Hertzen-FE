#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programa para resolver una viga de Timoshenko por el método de los
elementos finitos.

Creado por: Alejandro Hincapié G
"""

import pandas as pd
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt

# %% defino las constantes y variables
Y = 0
TH = 1
#filename = 'viga_Uribe_Escamilla_ej_5_5'
#filename = 'viga_con_resortes'
filename = 'viga'


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
T     = (df['LaG_EI_q'][['EF', 'NL1', 'NL2', 'E', 'I', 'G', 'q1e', 'q2e', 'Aast']]
         .fillna(0).set_index('EF').sort_index())

LaG   = T[['NL1','NL2']].to_numpy() - 1   # Relación de cada EF con sus nodos
E     = T['E'].to_numpy()                 # modulo de elasticidad E del EF
I     = T['I'].to_numpy()                 # momento de inercia Iz del EF
qq     = T[['q1e', 'q2e']].to_numpy()      # relación de las cargas distribuidas
Aast  = T['Aast'].to_numpy()
G     = T['G'].to_numpy()

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


# %% ensamblo la matriz de rigidez global y el vector de fuerzas nodales
#  equivalentes global para la viga de Euler-Bernoulli

idx = [None]*nef      # grados de libertad del elemento e

for e in range(nef):  # ciclo sobre todos los elementos finitos
    idx[e] = np.r_[gdl[LaG[e, 0]], gdl[LaG[e, 1]]]
    # algunas constantes para hacer el codigo mas legible
    EI_L = E[e]*I[e]/L[e];
    GAast_L = G[e]*Aast[e]/L[e];
    Le = L[e];
    
    # Matriz de rigidez de flexion del elemento e
    Kb = EI_L * np.array([[0,  0,  0,  0],
                          [0,  1,  0, -1],
                          [0,  0,  0,  0],
                          [0, -1,  0,  1]])
    
    # Matriz de rigidez de cortante del elemento e   
    p = nef; # numero de puntos de integracion (1 por EF)
    Ks = GAast_L * np.array([[ 1   ,  Le/2  ,  -1   ,  Le/2  ],
                             [ Le/2,  Le**2/4,  -Le/2,  Le**2/4],
                             [-1   , -Le/2  ,   1   , -Le/2  ],
                             [ Le/2,  Le**2/4,  -Le/2,  Le**2/4]])
    

    Ke = Kb + Ks;

    # vector de fuerzas nodales equivalentes (ver Kb_Ks_timoshenko_lineal.m)
    fe = np.array([ Le*(2*qq[e,0] + qq[e,1])/6,
                   0,
                   Le*(qq[e,0] + 2*qq[e,1])/6,
                   0])
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

# %%---------------VERIFICACION--------------------
for e in range(nef):
    Bbe = np.array([(6*xi)/Le**2, (3*xi - 1)/Le, -(6*xi)/Le**2, (3*xi + 1)/Le])
    ae = a[idx[e]]

    print(Bbe.T.shape, ae.shape)

#%%
## calculo de los momentos flectores
## (se calcula el momento en el centro de cada elemento finito)
# se reserva la memoria
# recuerde que en cada elemento se calculan los momentos en las raices de 
# los polinomios de Legendre de grado dos
xmom  = np.zeros(nef) # posicion donde se calcula momento flector
mom   = np.zeros(nef) # momento flector
xib   = 0        # raices del polinom de Legendre de grado 1 (vect. col)

xcor  = np.zeros(nef) # posicion donde se calcula fuerza cortante
cor   = np.zeros(nef) # fuerza cortante
xis   = 0       # raiz del polinomio de Legendre de grado 1 (vect. col)

for e in range(nef):
    Le = L[e]    
     
    # lugar donde se calcula el momento flector y la fuerza cortante
    # (centro del EF)
    xmom[e] = (Le*xib/2 + (xnod[LaG[e,0]] + xnod[LaG[e,1]])/2)
    xcor[e] = (Le*xis/2 + (xnod[LaG[e,0]] + xnod[LaG[e,1]])/2)

    ae = a[idx[e]]
    Bb = np.array([0, -1, 0, 1])/Le  # matriz de deformacion de flexion
    kappa = Bb@ae                 # curvatura
    mom[e] = E[e]*I[e]*kappa    # momento flector
    
    # gamma_xz y fuerza cortante
    Bs = np.array([ -1/Le,  (xis-1)/2,  1/Le,  -(xis+1)/2 ])
    
    gxz = Bs@ae                   # gamma_xz  
    cor[e] = -Aast[e]*G[e]*gxz    # fuerza cortante   


# %% se calculan los desplazamientos al interior de cada EF
nint = 10                      # No. de puntos donde se interpolará en el EF
xi = np.linspace(-1, 1, nint)  # coordenadas naturales

xx = nef*[None]  # interpol de posiciones (geometria) en el elemento
ww = nef*[None]  # interpol desplazamientos en el elemento
tt = nef*[None]  # interpol angulo en el elemento

Nw = np.c_[ (1-xi)/2, np.zeros(nint), (1+xi)/2, np.zeros(nint)]
Nt = np.c_[np.zeros(nint), (1-xi)/2, np.zeros(nint), (1+xi)/2]

for e in range(nef):  # ciclo sobre todas los elementos finitos

    Le = L[e]  # longitud del elemento finito e

    # Matriz de funciones de forma y su derivada:


    # vector de desplazamientos nodales del elemento a^{(e)}
    ae = a[idx[e]]

    # interpola sobre la geometria (coord naturales a geometricas)
    xx[e] = Le*xi/2 + (xnod[LaG[e, 0]] + xnod[LaG[e, 1]])/2

    # se calcula el desplazamiento al interior del elemento finito
    ww[e] = Nw@ae

    # se calcula el angulo al interior del elemento finito
    tt[e] = Nt@ae


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

    ax3.plot(x, [mom[e], mom[e]], lw=1, color='red' if mom[e]<=0 else 'green')
    ax3.fill_between(x, [mom[e], mom[e]],
                     color='salmon' if mom[e]<=0 else 'lightgreen')
    # Fuerza cortante:
    ax4.plot(x, [cor[e], cor[e]], lw=1, color='red' if cor[e]<=0 else 'green')
    ax4.fill_between(x, [cor[e], cor[e]],
                     color='salmon' if cor[e]<=0 else 'lightgreen')

# %% Se exportan los resultados:

despl = a.reshape(nno,2)

np.savetxt('./resultados_Timo/despl_Timo.csv', despl)
np.savetxt('./resultados_Timo/M_Timo.csv', mom)
np.savetxt('./resultados_Timo/Q_Timo.csv', cor)
