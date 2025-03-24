#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programa para resolver una viga por el método de los elementos finitos,
utilizando el elemento finito propuesto por Kartunnen y Von Hertzen.

Creado por: Alejandro Hincapié G
"""

import pandas as pd
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
from paper_2d import *


def plot_momento(e, ax):
    ''' Grafica el momento flector para el elemento finito "e". La gráfica se
        realiza sobre el Axes especificado como "ax".
    '''
    x = xnod[LaG[e]]  # Coordenadas de los nodos del EF
    M = mom[e]      # Momentos en los nodos del EF

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
        ax.plot(x, mom[e], lw=1,
                 color='red' if np.all(mom[e] <= 0) else 'green')
        ax.fill_between(x, mom[e],
                     color='salmon' if np.all(mom[e] <= 0) else 'lightgreen')

# %% defino las constantes y variables
Y = 0
TH = 1

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
T     = (df['LaG_EI_q'][['EF', 'NL1', 'NL2', 'E', 'I', 'q1e', 'q2e', 'b', 'h', 'nu']]
         .fillna(0).set_index('EF').sort_index())

LaG   = T[['NL1','NL2']].to_numpy() - 1   # Relación de cada EF con sus nodos
E     = T['E'].to_numpy()                 # modulo de elasticidad E del EF
I     = T['I'].to_numpy()                 # momento de inercia Iz del EF
qq    = T[['q1e', 'q2e']].to_numpy()              # relación de las cargas distribuidas
bb    = T['b'].to_numpy()
hh    = T['h'].to_numpy()
nuu   = T['nu'].to_numpy()


# %% relacion de los apoyos
T       = (df['restric'][['nodo', 'direccion', 'desplazamiento']]
           .set_index('nodo').sort_index())
dirdesp = T['direccion'].to_numpy() - 1
ac      = T['desplazamiento'].to_numpy()  # desplazamientos conocidos
nod_r = T.index.unique().to_numpy() - 1  # Nodos restringidos horizontalmente

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

# %% Matriz de rigidez elemento barra:
Kr = np.zeros((nno, nno))
fr = np.zeros(nno)


# %% ensamblo la matriz de rigidez global y el vector de fuerzas nodales
#  equivalentes global para la viga:

idx = [None]*nef      # grados de libertad del elemento e

for e in range(nef):  # ciclo sobre todos los elementos finitos
    idx[e] = np.r_[gdl[LaG[e, 0]], gdl[LaG[e, 1]]]
    idx_r = LaG[e]
    Le = L[e]
    qe = -qq[e].mean() # Esto se hace porque se asume que q debe ser constante
    be = bb[e]
    h = hh[e]
    nu = nuu[e]
    Ae = be*h
    Phi = 3*h**2*(1+nu)/Le**2
    
    # Matriz de rigidez de flexion del elemento e    

    Ke = (E[e]*I[e]/(Le**3*(Phi + 1)))*np.array([
                                [ 12,             6*Le,  -12,             6*Le],
                                [6*Le,  Le**2*(Phi + 4), -6*Le, Le**2*(-Phi + 2)],
                                [-12,            -6*Le,   12,            -6*Le],
                                [6*Le, Le**2*(-Phi + 2), -6*Le,  Le**2*(Phi + 4)]])


    # vector de fuerzas nodales equivalentes de una carga trapezoidal:

    fe = -qe/2*np.array([Le,
                        Le**2/6 - h**2*nu/4 - h**2/10,
                        Le,
                        -Le**2/6 + h**2*nu/4 + h**2/10])
    

    # se ensambla la matriz de rigidez K y el vector de fuerzas nodales
    # equivalentes f
    K[np.ix_(idx[e], idx[e])] += Ke
    f[idx[e]] += fe
    
    # Matriz de rigidez axial Kr del elemento:
    
    Kre = (Ae*E[e]/Le)*np.array([[ 1, -1],
                                 [-1,  1]])
    
    fre = -(h*nu*qe/2)*np.array([ 1, -1])

    Kr[np.ix_(idx_r, idx_r)] += Kre
    fr[idx_r] += fre

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

# Se hace lo mismo para los desplazamientos del elemento barra:

c_r = nod_r  # Grados de libertad restringidos horizontalmente
d_r = np.setdiff1d(np.arange(nno), c_r)
arc = np.zeros_like(c_r)

Krcc = Kr[np.ix_(c_r, c_r)];  Krcd = Kr[np.ix_(c_r, d_r)]; frd = f[c_r]
Krdc = Kr[np.ix_(d_r, c_r)];  Krdd = Kr[np.ix_(d_r, d_r)]; frc = f[d_r]

ard = np.linalg.solve(Krdd, frc - Krdc@arc)    # calculo desplazamientos desconocidos
qrd = Krcc@arc + Krcd@ard - frd                # calculo fuerzas de equilibrio desc.
a_r = np.zeros(nno);  a_r[c_r] = arc;  a_r[d_r] = ard  # desplazamientos 
q_r = np.zeros(nno);  q_r[c_r] = qrd  # fuerzas nodales equivalentes


# =============================================================================
# %% calculo de los momentos flectores, cortantes y axiales:
# =============================================================================

mom = np.empty((nef, 2))
xi = np.array([-sqrt(1/3), sqrt(1/3)])  # raices del P. de Legendre de grado 2
cor = np.empty(nef)
nor = np.empty(nef)


for e in range(nef):

    Le = L[e]  # longitud del elemento finito e
    qe = -qq[e, 0]
    nu = nuu[e]
    Ae = bb[e]*hh[e]
    Ee = E[e]
    h = hh[e]
    Ie = I[e]

    aer = a_r[LaG[e]]  # desplazamientos axiales del elemento e        
    
    Bn = (Ae*Ee/Le)*np.array([-1, 1])
    pn = Ae*h**3*nu*qe/(24*Ie)

    nor[e] = Bn@aer + pn
    
    # vector de desplazamientos nodales del elemento a^{(e)}
    ae = a[idx[e]]

    Bb = (Ee*Ie/(Le**2 + 3*h**2*nu + 3*h**2)
         *np.array([[-6, -(4*Le**2 + 3*h**2*nu + 3*h**2)/Le,  6, (-2*Le**2 + 3*h**2*nu + 3*h**2)/Le],
                    [ 6,  (2*Le**2 - 3*h**2*nu - 3*h**2)/Le, -6,  (4*Le**2 + 3*h**2*nu + 3*h**2)/Le]]))


    mom[e,:] = Bb@ae  # momento flector

    Bs = (Ee*Ie/(Le**2 + 3*h**2*nu + 3*h**2))*np.array([12/Le, 6, -12/Le, 6])

    cor[e] = Bs@ae


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


# %% Se grafican los resultados:



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


despl = a.reshape(nno,2)

ax1.plot(xnod, despl[:,0])
ax2.plot(xnod, despl[:,1])

for e in range(nef):
    plot_momento(e, ax3)
    ax4.plot(xnod[LaG[e,:]], [cor[e], cor[e]],
             lw=1, color='red' if cor[e]<=0 else 'green')
    ax4.fill_between(xnod[LaG[e,:]], [cor[e], cor[e]],
                     color='salmon' if cor[e]<=0 else 'lightgreen')

for ax in (ax1, ax2, ax3, ax4):
    ax.axhline(color='k', lw=1)

# %% Se exportan los resultados:

np.savetxt('./resultados_paper/despl_paper.csv', despl)
np.savetxt('./resultados_paper/N_paper.csv', nor)
np.savetxt('./resultados_paper/M_paper.csv', mom)
np.savetxt('./resultados_paper/Q_paper.csv', cor)
np.savetxt('./resultados_paper/2d/a_paper.csv', a)
np.savetxt('./resultados_paper/2d/ar_paper.csv', a_r)

# %% Se calcula y grafica el campo de desplazamientos bidimensionales: 

h = hh[0]

fig3, ax5 = plt.subplots(figsize=(19, 6))
escala = 100
for e in range(nef):
    plot_despl(e, LaG, xnod, qq, L, nuu, E, hh, I, idx, a, a_r, ax5, escala)
for x in [0, 5]:
    ax5.plot([x, x], [-h/2, h/2], lw=1, c='b')
ax5.set_aspect('equal', adjustable='box')
ax5.set_title(f'Desplazamientos 2D aumentados {escala} veces')
ax5.set_yticks(np.linspace(-h/2, h/2, 5))
ax5.set_xlabel('x [m]')
ax5.set_ylabel('y [m]')

