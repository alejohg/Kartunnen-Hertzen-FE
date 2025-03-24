# -*- coding: utf-8 -*-

# %%
'''
-------------------------------------------------------------------------------
NOTA: este código SOLO es apropiado para TENSION PLANA usando elementos
      triangulares de 10 nodos.
-------------------------------------------------------------------------------
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from funciones import t2ft_T10, compartir_variables, plot_esf_def, plot_esf_def_clip
from TriGauss import TriGaussPoints
from leer_GMSH import xnod_from_msh, LaG_from_msh

# %% constantes que ayudarán en la lectura del código
X, Y = 0, 1
NL1, NL2, NL3, NL4, NL5, NL6, NL7, NL8, NL9, NL10 = range(10)


# %% defino las variables/constantes del sólido
Ee   = 3.5e7  # [KPa]   módulo de elasticidad del sólido
nue  = 0.30   # [-]     coeficiente de Poisson
rhoe = 0      # [T/m³]  densidad
te   = 0.20   # [m]     espesor del sólido
g    = 0      # [m/s²]  aceleración de la gravedad

# %% Malla a utilizar:

# 2) Malla refinada (malla elaborada en gmsh)
malla = './malla/malla.msh'  # Archivo exportado por GMSH
malla_excel = './malla/malla.xlsx' # Archivo excel con datos varios

# %% posición de los nodos:
# xnod: fila=número del nodo, columna=coordenada X=0 o Y=1
xnod = xnod_from_msh(malla)  # Se halla la matriz con la función respectiva
nno  = xnod.shape[0]      # número de nodos (número de filas de la matriz xnod)

# %% definición de los grados de libertad
ngdl = 2*nno            # número de grados de libertad (dos por nodo)
gdl  = np.reshape(np.arange(ngdl), (nno,2)) # nodos vs grados de libertad

# %% definición de elementos finitos con respecto a nodos
# LaG: fila=número del elemento, columna=número del nodo local
LaG = LaG_from_msh(malla)
nef = LaG.shape[0]      # número de EFs (número de filas de la matriz LaG)

# %% Relación de cargas puntuales
cp  = pd.read_excel(malla_excel, sheet_name='carga_punt')
ncp = cp.shape[0]       # número de cargas puntuales
f   = np.zeros(ngdl)    # vector de fuerzas nodales equivalentes global
for i in range(ncp):
   f[gdl[cp['nodo'][i]-1, cp['dirección'][i]-1]] = cp['fuerza puntual'][i]


# %% Se dibuja la malla de elementos finitos:

cg = np.zeros((nef,2))  # almacena el centro de gravedad de los EF

plt.figure(figsize=(6,6))
for e in range(nef):
   nod_ef = LaG[e, [NL1, NL4, NL5, NL2, NL6, NL7, NL3, NL8, NL9, NL1]]
   plt.plot(xnod[nod_ef, X], xnod[nod_ef, Y], 'b')

   # se calcula la posición del centro de gravedad del triángulo
   cg[e] = np.mean(xnod[LaG[e,:], :], axis=0)

   # y se reporta el número del elemento actual
#   plt.text(cg[e,X], cg[e,Y], f'{e+1}', horizontalalignment='center',
#                                        verticalalignment='center',  color='b')

#plt.plot(xnod[:,X], xnod[:,Y], 'r.')
#for i in range(nno):
#   plt.text(xnod[i,X], xnod[i,Y], f'{i+1}', color='r')
plt.axis('Equal')  
plt.tight_layout()
plt.title('Malla de elementos finitos')
plt.show()


# %% Funciones de forma y sus derivadas del elemento triangular de 10 nodos:

def Nforma(xi, eta):
    ''' Evalúa las funciones de forma en un punto determinado de coordenadas
        (xi, eta).
        Funciones de forma obtenidas del programa FF_triangulo.py.
    '''

    L1 = 1 - xi - eta
    L2 = xi
    L3 = eta

    N = np.array(
                [L1*(4.5*L1**2 - 4.5*L1 + 1.0),
                 L2*(4.5*L2**2 - 4.5*L2 + 1.0),
                 L3*(4.5*L3**2 - 4.5*L3 + 1.0),
                 L1*L2*(13.5*L1 - 4.5),
                 L1*L2*(13.5*L2 - 4.5),
                 L2*L3*(13.5*L2 - 4.5),
                 L2*L3*(13.5*L3 - 4.5),
                 L1*L3*(13.5*L3 - 4.5),
                 L1*L3*(13.5*L1 - 4.5),
                 27.0*L1*L2*L3       ])
    return N


def dN_dxi(xi, eta):
    ''' Evalúa las derivadas de las funciones de forma respecto a xi en un
        punto determinado de coordenadas (xi, eta). 
        Derivadas obtenidas del programa FF_triangulo.py.
    '''

    L2 = xi
    L3 = eta

    dN_dL2 = np.array([
            -13.5*L2**2 - 27.0*L2*L3 + 18.0*L2 - 13.5*L3**2 + 18.0*L3 - 5.5,
            13.5*L2**2 - 9.0*L2 + 1.0,
            0,
            40.5*L2**2 + 54.0*L2*L3 - 45.0*L2 + 13.5*L3**2 - 22.5*L3 + 9.0,
            -40.5*L2**2 - 27.0*L2*L3 + 36.0*L2 + 4.5*L3 - 4.5,
            L3*(27.0*L2 - 4.5),
            L3*(13.5*L3 - 4.5),
            L3*(-13.5*L3 + 4.5),
            L3*(27.0*L2 + 27.0*L3 - 22.5),
            27.0*L3*(-2*L2 - L3 + 1)
            ])

    return dN_dL2


def dN_deta(xi, eta):
    ''' Evalúa las derivadas de las funciones de forma respecto a eta en un
        punto determinado de coordenadas (xi, eta).
        Derivadas obtenidas del programa FF_triangulo.py.
    '''

    L2 = xi
    L3 = eta

    dN_dL3 = np.array([
            -13.5*L2**2 - 27.0*L2*L3 + 18.0*L2 - 13.5*L3**2 + 18.0*L3 - 5.5,
            0,
            13.5*L3**2 - 9.0*L3 + 1.0,
            L2*(27.0*L2 + 27.0*L3 - 22.5),
            L2*(-13.5*L2 + 4.5),
            L2*(13.5*L2 - 4.5),
            L2*(27.0*L3 - 4.5),
            -27.0*L2*L3 + 4.5*L2 - 40.5*L3**2 + 36.0*L3 - 4.5,
            13.5*L2**2 + 54.0*L2*L3 - 22.5*L2 + 40.5*L3**2 - 45.0*L3 + 9.0,
            27.0*L2*(-L2 - 2*L3 + 1)
            ])

    return dN_dL3


#%% Cuadratura de Gauss-Legendre
# NOTA: se asumirá aquí el mismo orden de la cuadratura tanto en la dirección
# de xi como en la dirección de eta
n = 4  # orden de la cuadratura de Gauss-Legendre (para triángulos)
xw = TriGaussPoints(n)
xi_gl, eta_gl, w_gl = xw[:,0], xw[:,1], xw[:,2]
n_gl = xi_gl.size  # Número de puntos de Gauss.

# %% Ensamblaje la matriz de rigidez global y el vector de fuerzas másicas
#    nodales equivalentes global

# se inicializan la matriz de rigidez global y los espacios en memoria que
#  almacenarán las matrices de forma y de deformación
K = np.zeros((ngdl, ngdl))      # matriz de rigidez global
N = np.empty((nef,n_gl,2,2*10)) # matriz de forma en cada punto de GL
B = np.empty((nef,n_gl,3,2*10)) # matriz de deformaciones en cada punto de GL
idx = nef * [None]              # indices asociados a los gdl del EF e

# matriz constitutiva del elemento para TENSION PLANA
De = np.array([[Ee/(1 - nue**2),        Ee*nue/(1 - nue**2),    0               ],
               [Ee*nue/(1 - nue**2),    Ee/(1 - nue**2),        0               ],
               [0,                      0,                      Ee/(2*(1 + nue))]])
be = np.array([0, -rhoe*g])  # [kN/m³] vector de fuerzas másicas

# para cada elemento finito en la malla:
for e in range(nef):
    # se calculan con el siguiente ciclo las matrices de rigidez y el vector de
    # fuerzas nodales equivalentes del elemento usando las cuadraturas de GL
    Ke = np.zeros((2*10, 2*10))
    fe = np.zeros(2*10)
    det_Je = np.empty((n_gl))      # matriz para almacenar los jacobianos

    for i in range(n_gl):
        # en cada punto de la cuadratura de Gauss-Legendre se evalúan las
        # funciones de forma y sus derivadas
        xi, eta = xi_gl[i], eta_gl[i]

        NNforma  = Nforma (xi, eta)
        ddN_dxi  = dN_dxi (xi, eta)
        ddN_deta = dN_deta(xi, eta)

        # se llaman las coordenadas nodales del elemento para calcular las
        # derivadas de la función de transformación
        xe, ye = xnod[LaG[e], X], xnod[LaG[e], Y]

        dx_dxi  = np.sum(ddN_dxi  * xe);    dy_dxi  = np.sum(ddN_dxi  * ye)
        dx_deta = np.sum(ddN_deta * xe);    dy_deta = np.sum(ddN_deta * ye)

        # con ellas se ensambla la matriz Jacobiana del elemento y se
        # calcula su determinante
        Je = np.array([[dx_dxi,  dy_dxi ],
                       [dx_deta, dy_deta]])
        det_Je[i] = np.linalg.det(Je)

        # las matrices de forma y de deformación se evalúan y se ensamblan
        # en el punto de Gauss
        Npq = np.empty((2, 2*10))
        Bpq = np.empty((3, 2*10))
        
        for j in range(10):
            Npq[:,[2*j, 2*j+1]] = np.array([[NNforma[j], 0         ],
                                            [0,          NNforma[j]]])

            dNi_dx = (+dy_deta*ddN_dxi[j] - dy_dxi*ddN_deta[j])/det_Je[i]
            dNi_dy = (-dx_deta*ddN_dxi[j] + dx_dxi*ddN_deta[j])/det_Je[i]
            Bpq[:,[2*j, 2*j+1]] = np.array([[dNi_dx, 0     ],
                                            [0,      dNi_dy],
                                            [dNi_dy, dNi_dx]])
        N[e,i] = Npq
        B[e,i] = Bpq

        # se ensamblan la matriz de rigidez del elemento y el vector de
        # fuerzas nodales equivalentes del elemento
        Ke += Bpq.T @ De @ Bpq * det_Je[i]*te*w_gl[i]
        fe += Npq.T @ be       * det_Je[i]*te*w_gl[i]

    # se determina si hay puntos con jacobiano negativo, en caso tal se termina
    # el programa y se reporta
    if np.any(det_Je <= 0):
        raise Exception(f'Hay puntos con det_Je negativo en el elemento {e+1}')

    # y se añaden la matriz de rigidez del elemento y el vector de fuerzas
    # nodales del elemento a sus respectivos arreglos de la estructura
    idx[e] = gdl[LaG[e]].flatten() # se obtienen los grados de libertad
    K[np.ix_(idx[e], idx[e])] += Ke
    f[np.ix_(idx[e])]         += fe


# %% Cálculo de las cargas nodales equivalentes de las cargas distribuidas:

cd   = pd.read_excel('./malla/superficiales.xlsx', sheet_name='carga_distr')
nlcd = cd.shape[0]     # número de lados con carga distribuida
ft   = np.zeros(ngdl)  # fuerzas nodales equivalentes de cargas superficiales

# por cada lado cargado se obtienen las fuerzas nodales equivalentes en los
# nodos y se añaden al vector de fuerzas superficiales
for i in range(nlcd):
   e     = cd['elemento'][i] - 1
   lado  = cd['lado'][i]
   carga = cd[['tix', 'tiy', 'tjx', 'tjy', 'tkx', 'tky', 'tlx', 'tly']].loc[i].to_numpy()
   fte   = t2ft_T10(xnod[LaG[e,:],:], lado, carga, te)

   ft[np.ix_(idx[e])] += fte

# %% agrego al vector de fuerzas nodales equivalentes las fuerzas
# superficiales calculadas
f += ft

# %% restricciones y los grados de libertad del desplazamiento conocidos (c)
restric = pd.read_excel(malla_excel, sheet_name='restric')
nres = restric.shape[0]
c    = np.empty(nres, dtype=int)
for i in range(nres):
   c[i] = gdl[restric['nodo'][i]-1, restric['dirección'][i]-1]

# desplazamientos conocidos
ac = restric['desplazamiento'].to_numpy()

# grados de libertad del desplazamiento desconocidos
d = np.setdiff1d(range(ngdl), c)

# %% extraigo las submatrices y especifico las cantidades conocidas
# f = vector de fuerzas nodales equivalentes
# q = vector de fuerzas nodales de equilibrio del elemento
# a = desplazamientos

#| qd |   | Kcc Kcd || ac |   | fd |  # recuerde que qc=0 (siempre)
#|    | = |         ||    | - |    |
#| qc |   | Kdc Kdd || ad |   | fc |
Kcc = K[np.ix_(c,c)];  Kcd = K[np.ix_(c,d)]; fd = f[c]
Kdc = K[np.ix_(d,c)];  Kdd = K[np.ix_(d,d)]; fc = f[d]

# %% resuelvo el sistema de ecuaciones
ad = np.linalg.solve(Kdd, fc - Kdc@ac) # desplazamientos desconocidos
qd = Kcc@ac + Kcd@ad - fd              # fuerzas de equilibrio desconocidas

# armo los vectores de desplazamientos (a) y fuerzas (q)
a = np.zeros(ngdl); q = np.zeros(ngdl) # separo la memoria
a[c] = ac;          a[d] = ad          # desplazamientos
q[c] = qd         # q[d] = qc = 0      # fuerzas nodales de equilibrio

# %% Dibujo la malla de elementos finitos y las deformada de esta
delta  = np.reshape(a, (nno,2))
escala = 100              # factor de escalamiento de la deformada
xdef   = xnod + escala*delta    # posición de la deformada

plt.figure(figsize=(6, 6))
for e in range(nef):
   nod_ef = LaG[e, [NL1, NL4, NL5, NL2, NL6, NL7, NL3, NL8, NL9, NL1]]
   plt.plot(xnod[nod_ef, X], xnod[nod_ef, Y], 'r--',
                        label='Posición original'  if e == 0 else "", lw=0.5)
   plt.plot(xdef[nod_ef, X], xdef[nod_ef, Y], 'b',
                        label='Posición deformada' if e == 0 else "", lw=1)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.title(f'Deformada escalada {escala} veces')
plt.tight_layout()
plt.savefig('./resultados_TP/deformada2.eps')
plt.show()

#%% Deformaciones y los esfuerzos en los puntos de Gauss
deform = np.zeros((nef,n_gl,3)) # deformaciones en cada punto de GL
esfuer = np.zeros((nef,n_gl,3)) # esfuerzos en cada punto de GL

for e in range(nef):
    ae = a[idx[e]]    # desplazamientos nodales del elemento e
    for i in range(n_gl):
            deform[e,i] = B[e,i] @ ae       # calculo las deformaciones
            esfuer[e,i] = De @ deform[e,i]  # calculo los esfuerzos

#%% Esfuerzos y deformaciones en los nodos:
num_elem_ady = np.zeros(nno)
sx  = np.zeros(nno);    txy = np.zeros(nno);    ex  = np.zeros(nno)
sy  = np.zeros(nno);    txz = np.zeros(nno);    ey  = np.zeros(nno)
sz  = np.zeros(nno);    tyz = np.zeros(nno);    gxy = np.zeros(nno)

# matriz de extrapolación

A = np.array([
       [ 0.12634073, -0.63855959, -0.63855959,  1.87365927,  0.13855959,  0.13855959],
       [-0.63855959, -0.63855959,  0.12634073,  0.13855959,  0.13855959,  1.87365927],
       [-0.63855959,  0.12634073, -0.63855959,  0.13855959,  1.87365927,  0.13855959],
       [-0.20780502,  1.13839679, -0.4627718 ,  0.46755195,  0.17544269, -0.11081461],
       [-0.4627718 ,  1.13839679, -0.20780502, -0.11081461,  0.17544269,  0.46755195],
       [ 1.13839679, -0.4627718 , -0.20780502,  0.17544269, -0.11081461,  0.46755195],
       [ 1.13839679, -0.20780502, -0.4627718 ,  0.17544269,  0.46755195, -0.11081461],
       [-0.4627718 , -0.20780502,  1.13839679, -0.11081461,  0.46755195,  0.17544269],
       [-0.20780502, -0.4627718 ,  1.13839679,  0.46755195, -0.11081461,  0.17544269],
       [ 0.42570639,  0.42570639,  0.42570639, -0.09237306, -0.09237306, -0.09237306]])
# se hace la extrapolación de los esfuerzos y las deformaciones en cada elemento
# a partir de las lecturas en los puntos de Gauss
for e in range(nef):
    #sx[LaG[e]]  += A @ np.array([esfuer[e,0,0,0],   # I   = (p=0, q=0)
    #                             esfuer[e,0,1,0],   # II  = (p=0, q=1)
    #                             esfuer[e,1,0,0],   # III = (p=1, q=0)
    #                             esfuer[e,1,1,0]])  # IV  = (p=1, q=1)
    sx [LaG[e]] += A @ esfuer[e,:,0].ravel()
    sy [LaG[e]] += A @ esfuer[e,:,1].ravel()
    txy[LaG[e]] += A @ esfuer[e,:,2].ravel()
    ex [LaG[e]] += A @ deform[e,:,0].ravel()
    ey [LaG[e]] += A @ deform[e,:,1].ravel()
    gxy[LaG[e]] += A @ deform[e,:,2].ravel()

    # se lleva un conteo de los elementos adyacentes a un nodo
    num_elem_ady[LaG[e]] += 1

# en todos los nodos se promedia los esfuerzos y las deformaciones de los
# elementos, se alisa la malla de resultados
sx  /= num_elem_ady;   ex  /= num_elem_ady
sy  /= num_elem_ady;   ey  /= num_elem_ady
txy /= num_elem_ady;   gxy /= num_elem_ady

# se calculan las deformaciones ez en tension plana
ez = -(nue/Ee)*(sx + sy)

# %% Se calculan y grafican para cada elemento los esfuerzos principales y
#    sus direcciones
# NOTA: esto solo es valido para el caso de TENSION PLANA).
# En caso de DEFORMACIÓN PLANA se deben calcular los valores y vectores
# propios de la matriz de tensiones de Cauchy
#   [dirppales{e}, esfppales{e}] = eig([sx  txy 0    % matriz de esfuerzos
#                                       txy sy  0    % de Cauchy
#                                       0   0   0]);
s1   = (sx+sy)/2 + np.sqrt(((sx-sy)/2)**2 + txy**2) # esfuerzo normal máximo
s2   = (sx+sy)/2 - np.sqrt(((sx-sy)/2)**2 + txy**2) # esfuerzo normal mínimo
tmax = (s1 - s2)/2                                  # esfuerzo cortante máximo
ang  = 0.5*np.arctan2(2*txy, sx-sy)                 # ángulo de inclinación de s1

# %% Calculo de los esfuerzos de von Mises
s3 = np.zeros(nno)
sv = np.sqrt(((s1-s2)**2 + (s2-s3)**2 + (s1-s3)**2)/2)

# %% Gráfica del post-proceso:
# las matrices xnod y LaG se vuelven variables globales por facilidad
# =============================================================================
# compartir_variables(xnod, LaG)
# 
# # deformaciones
# plot_esf_def(ex,   r'$\epsilon_x$', nombre='./resultados_TP/ex2.eps')
# plot_esf_def(ey,   r'$\epsilon_y$', nombre='./resultados_TP/ey2.eps')
# plot_esf_def(ez,   r'$\epsilon_z$', nombre='./resultados_TP//ez2.eps')
# plot_esf_def(gxy,  r'$\gamma_{xy}$ [rad]', nombre='./resultados_TP/gxy2.eps')
# 
# # esfuerzos
# plot_esf_def(sx,   r'$\sigma_x$ [kPa]', nombre='./resultados_TP/sx2.eps')
# plot_esf_def(sy,   r'$\sigma_y$ [kPa]', nombre='./resultados_TP/sy2.eps')
# plot_esf_def(txy,  r'$\tau_{xy}$ [kPa]', nombre='./resultados_TP/txy2.eps')
# 
# # esfuerzos principales con sus orientaciones
# plot_esf_def(s1,   r'$\sigma_1$ [kPa]', ang, nombre='./resultados_TP/s12.eps')
# plot_esf_def(s2,   r'$\sigma_2$ [kPa]', ang+np.pi/2,
#              nombre='./resultados_TP/s22.eps')
# plot_esf_def(tmax, r'$\tau_{máx}$ [kPa]', [ ang-np.pi/4, ang+np.pi/4 ],
#              nombre='./resultados_TP/tmax2.eps')
# 
# # esfuerzos de von Mises
# plot_esf_def(sv,   r'$\sigma_{VM}$ [kPa]', nombre='./resultados_TP/sv2.eps')
# =============================================================================


# %% Reporte de los resultados:

# se crean tablas para reportar los resultados nodales de: desplazamientos (a),
# fuerzas nodales equivalentes (f) y fuerzas nodales de equilibrio (q)
tabla_desp = pd.DataFrame(
    data=np.c_[xnod, a.reshape((nno,2))],
    index=np.arange(nno)+1,
    columns=['x', 'y', 'ux', 'uy'])
tabla_desp.index.name = '# nodo'

 # esfuerzos
tabla_esf = pd.DataFrame(data    = np.c_[xnod, sx, txy],
                          index   = np.arange(nno) + 1,
                          columns = ['x', 'y', 'sx', 'txy'])
tabla_esf.index.name = '# nodo'
# =============================================================================
# 
# # deformaciones
# tabla_def = pd.DataFrame(data    = np.c_[ex, ey, ez, gxy],
#                          index   = np.arange(nno) + 1,
#                          columns = ['ex', 'ey', 'ez', 'gxy [rad]'])
# tabla_def.index.name = '# nodo'

# 
# # esfuerzos principales y de von Misses:
# tabla_epv = pd.DataFrame(
#        data    = np.c_[s1, s2, tmax, sv, ang],
#        index   = np.arange(nno) + 1,
#        columns = ['s1 [kPa]', 's2 [kPa]', 'tmax [kPa]', 'sv [kPa]', 'theta [krad]'])
# tabla_epv.index.name = '# nodo'
# 
# # se crea un archivo de MS EXCEL
# nombre_archivo = './Resultados malla gmsh/resultados.xlsx'
# writer = pd.ExcelWriter(nombre_archivo, engine = 'xlsxwriter')
# 
# # cada tabla hecha previamente es guardada en una hoja del archivo de Excel
# tabla_afq.to_excel(writer, sheet_name = 'afq')
# tabla_def.to_excel(writer, sheet_name = 'exeyezgxy')
# tabla_esf.to_excel(writer, sheet_name = 'sxsytxy')
# tabla_epv.to_excel(writer, sheet_name = 's1s2tmaxsv')
# writer.save()
# 
# print(f'Cálculo finalizado. En "{nombre_archivo}" se guardaron los resultados.')
# =============================================================================


# =============================================================================
# # %% Se grafican los resultados pero realizando un clipping a los datos, es de-
# # cir, reduciendo los valores mínimo y máximo.
# 
# compartir_variables(xnod, LaG)
# 
# # deformaciones
# plot_esf_def_clip(ex,   r'$\epsilon_x$', nombre='./graficos gmsh/ex2_clip.eps')
# plot_esf_def_clip(ey,   r'$\epsilon_y$', nombre='./graficos gmsh/ey2_clip.eps')
# plot_esf_def_clip(ez,   r'$\epsilon_z$', nombre='./graficos gmsh/ez2_clip.eps')
# plot_esf_def_clip(gxy,  r'$\gamma_{xy}$ [rad]', nombre='./graficos gmsh/gxy2_clip.eps')
# 
# # esfuerzos
# plot_esf_def_clip(sx,   r'$\sigma_x$ [kPa]', nombre='./graficos gmsh/sx2_clip.eps')
# plot_esf_def_clip(sy,   r'$\sigma_y$ [kPa]', nombre='./graficos gmsh/sy2_clip.eps')
# plot_esf_def_clip(txy,  r'$\tau_{xy}$ [kPa]', nombre='./graficos gmsh/txy2_clip.eps')
# 
# # esfuerzos principales con sus orientaciones
# plot_esf_def_clip(s1,   r'$\sigma_1$ [kPa]', ang, nombre='./graficos gmsh/s12_clip.eps')
# plot_esf_def_clip(s2,   r'$\sigma_2$ [kPa]', ang+np.pi/2,
#                   nombre='./graficos gmsh/s22_clip.eps')
# plot_esf_def_clip(tmax, r'$\tau_{máx}$ [kPa]', [ ang-np.pi/4, ang+np.pi/4 ],
#              nombre='./graficos gmsh/tmax2_clip.eps')
# 
# # esfuerzos de von Mises
# plot_esf_def_clip(sv,   r'$\sigma_{VM}$ [kPa]', nombre='./graficos gmsh/sv2_clip.eps')
# 
# =============================================================================

# %% Se exporta el vector de desplazamientos:

np.savetxt('./resultados_TP/a_tp.csv', a)
           
# %% Se exportan los resultados de desplazamientos en el eje neutro:

h = xnod[:,1].max()

res_en = tabla_desp[tabla_desp['y']==h/2].sort_values('x') # resultados eje neutro

x_tp = res_en['x']
w_tp = res_en['uy']

np.savetxt('./resultados_TP/x_tp.csv', x_tp)
np.savetxt('./resultados_TP/w_tp.csv', w_tp)


# %% Se exportan resultados de esfuerzos en la sección x=x0

x0 = 2.5  # Coordenada x de la sección que se quiere analizar

res_3 = tabla_esf[tabla_esf['x'] == x0].sort_values('y')

y_3 = res_3['y']
sx_3 = res_3['sx']
txy_3 = res_3['txy']

res_3.to_csv('./resultados_TP/res_esf3.csv')

# %% Se exportan resultados de desplazamientos en x=x0:

res_d3 = tabla_desp[tabla_desp['x'] == x0].sort_values('y') # En m

res_d3['x_d'] = tabla_desp['x'] + tabla_desp['ux']
res_d3['y_d'] = tabla_desp['y'] + tabla_desp['uy']

res_d3.to_csv('./resultados_TP/res_d3.csv')

# %%bye, bye!