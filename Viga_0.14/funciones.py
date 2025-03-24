# -*- coding: utf-8 -*-

import numpy                as np
import matplotlib.pyplot    as plt
import warnings
from numpy.polynomial.legendre import leggauss


# %% constantes que ayudaran en la lectura del codigo
X, Y = 0, 1
NL1, NL2, NL3, NL4, NL5, NL6, NL7, NL8, NL9, NL10 = range(10)

# %% variables globales que se heredarán del programa principal
xnod = None
LaG = None

# %%
def compartir_variables(xnod_, LaG_):
    '''
    Importa variables globales del programa principal a este módulo
    '''
    global xnod, LaG
    xnod, LaG = xnod_, LaG_
        
# %%
def t2ft_T10(xnod, lado, carga, espesor):
    '''Función que convierte las fuerzas superficiales aplicadas a un elemento
    finito triangular de 10 nodos a sus correspondientes cargas nodales equiva-
    lentes ft

    Recibe:
        xnod:  coordenadas nodales del elemento finito

        xnod = [ x1e y1e
                x2e  y2e
                 ... ...
                x8e  y8e ]

        lado:  arista en la que se aplica la carga, puede tomar los siguientes
               valores: 1452, 2673, 3891

        carga: fuerza distribuida en los nodos
        
    
        espesor: espesor del elemento
    '''
    
    # se definen los indices de los lados
    if   lado == 1452: idx = np.array([ 1, 4, 5, 2 ]) - 1
    elif lado == 2673: idx = np.array([ 2, 6, 7, 3 ]) - 1
    elif lado == 3891: idx = np.array([ 3, 8, 9, 1 ]) - 1
    else: 
        raise Exception('Únicamente se permiten los lados 1452, 2673 o 3891')

    nno = xnod.shape[0]
    if nno is not 10:
        raise Exception('Solo para elementos triangulares de 10 nodos')

    # parámetros para mejorar la lectura del código
    X, Y = 0, 1
   
    # se define el número de puntos de la cuadratura y se obtienen los puntos
    # de evaluación y los pesos de la cuadratura

    n_gl       = 4
    xi_gl, w_gl = leggauss(n_gl)
    
    
    # se definen las funciones de forma unidimensionales y sus derivadas
    NN     = lambda xi: np.array([-9*xi**3/16 + 9*xi**2/16 + xi/16 - 1/16, 
                                   27*xi**3/16 - 9*xi**2/16 - 27*xi/16 + 9/16,
                                  -27*xi**3/16 - 9*xi**2/16 + 27*xi/16 + 9/16,
                                    9*xi**3/16 + 9*xi**2/16 - xi/16 - 1/16])

    dNN_dxi = lambda xi: np.array([-27*xi**2/16 + 9*xi/8 + 1/16,
                                     81*xi**2/16 - 9*xi/8 - 27/16,
                                    -81*xi**2/16 - 9*xi/8 + 27/16,
                                     27*xi**2/16 + 9*xi/8 - 1/16])

    # se calcula el vector de fuerzas distribuidas en los nodos
    te = np.zeros(2*nno)
    te[np.c_[2*idx, 2*idx + 1].ravel()] = carga

    # cálculo de la integral:
    suma   = np.zeros((2*nno, 2*nno))
    N      = np.zeros(nno)
    dN_dxi = np.zeros(nno)
    for p in range(n_gl):
        # se evalúan las funciones de forma
        N[idx] = NN(xi_gl[p])
        matN = np.empty((2,2*nno))
        for i in range(nno):
            matN[:,[2*i, 2*i+1]] = np.array([[N[i], 0   ],
                                             [0,    N[i]]])

        # se calcula el jacobiano
        dN_dxi[idx] = dNN_dxi(xi_gl[p])
        dx_dxi      = np.dot(dN_dxi, xnod[:,X])
        dy_dxi      = np.dot(dN_dxi, xnod[:,Y])
        ds_dxi      = np.hypot(dx_dxi, dy_dxi)

        # y se calcula la sumatoria
        suma += matN.T @ matN * ds_dxi*w_gl[p]

    # finalmente, se retorna el vector de fuerzas nodales equivalentes
    return espesor * (suma @ te)

#%%
def plot_esf_def(variable, titulo, angulo = None, nombre=None):
    '''
    Grafica variable para la malla de EFs especificada.

    Uso:
        variable: es la variable que se quiere graficar
        titulo:   título del gráfico
        angulo:   opcional para los esfuerzos principales s1 y s2 y tmax
        nombre:   nombre del gráfico para guardarlo en formato .eps

    Para propósitos de graficación el EF se divide en 6 triángulos así: 

                           eta
                            ^
                            |
                            3
                            |\
                            | \
                            |  \
                            |(9)\
                            8----7
                            |(6)/|\
                            |  / | \
                            | /  |  \
                            |/(7)|(8)\
                            9----10---6
                            |\(2)|(3)/|\
                            | \  |  / | \
                            |  \ | /  |  \
                            |(1)\|/(4)|(5)\
                            1----4----5----2-----> xi

    El número entre paréntesis indica el número de cada EF triangular pequeño.                        
    '''
    
    nef = LaG.shape[0]    # número de elementos finitos en la malla original

    # se arma la matriz de correspondencia (LaG) de la nueva malla triangular
    LaG_t = np.zeros((9*nef, 3), dtype = int)
    for e in range(nef):
        LaG_t[9*e + 0, :] = LaG[e, [NL1,  NL4,  NL9 ]]
        LaG_t[9*e + 1, :] = LaG[e, [NL4,  NL10, NL9 ]]
        LaG_t[9*e + 2, :] = LaG[e, [NL4,  NL6,  NL10]]
        LaG_t[9*e + 3, :] = LaG[e, [NL4,  NL5,  NL6 ]]
        LaG_t[9*e + 4, :] = LaG[e, [NL5,  NL2,  NL6 ]]
        LaG_t[9*e + 5, :] = LaG[e, [NL9,  NL7,  NL8 ]]
        LaG_t[9*e + 6, :] = LaG[e, [NL9,  NL10, NL7 ]]
        LaG_t[9*e + 7, :] = LaG[e, [NL10, NL6,  NL7 ]]
        LaG_t[9*e + 8, :] = LaG[e, [NL8,  NL7,  NL3 ]]

    
    # se inicializa el lienzo
    fig, ax = plt.subplots(figsize=(7,7)) 

    # se encuentra el máximo en valor absoluto para ajustar el colorbar()
    val_max = np.max(np.abs(variable))  

    # se grafica la malla de EFS, los colores en cada triángulo y las curvas 
    # de nivel
    for e in range(nef):
        # se dibujan las aristas
        nod_ef = LaG[e, [NL1, NL4, NL5, NL2, NL6, NL7, NL3, NL8, NL9, NL1]]
        plt.plot(xnod[nod_ef, X], xnod[nod_ef, Y], lw = 0.5, color = 'gray')

    im = ax.tripcolor(xnod[:, X], xnod[:, Y], LaG_t, variable, cmap = 'bwr',
                      shading = 'gouraud', vmin = -val_max, vmax = val_max)
    ax.tricontour(xnod[:, X], xnod[:, Y], LaG_t, variable, 20)
    
    # a veces sale un warning, simplemente porque no existe la curva 0
    warnings.filterwarnings("ignore")
    ax.tricontour(xnod[:, X], xnod[:, Y], LaG_t, variable, levels=[0], linewidths=3)
    warnings.filterwarnings("default")

    fig.colorbar(im, ax = ax, format = '%6.3g')
                
    # para los esfuerzos principales se grafican las líneas que indiquen las
    # direcciones de los esfuerzos en cada nodo de la malla
    if angulo is not None:
       if type(angulo) is np.ndarray: 
           angulo = [ angulo ]
       for ang in angulo:
           ax.quiver(xnod[:, X], xnod[:, Y], 
                variable*np.cos(ang), variable*np.sin(ang), 
                headwidth=0, headlength=0, headaxislength=0, pivot='middle')
    
    # se especifican los ejes y el título, y se colocan los ejes iguales
    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')
    ax.set_title(titulo, fontsize=20)

    ax.set_aspect('equal')
    ax.autoscale(tight=True)
    plt.tight_layout()
    plt.savefig(nombre)
    plt.show()


#%%
def plot_esf_def_clip(variable, titulo, angulo = None, nombre=None):
    '''
    Grafica variable para la malla de EFs especificada, realizando un clipping,
    recortando la escala de valores, reduciendo el valor máximo dividiendolo
    entre un número dado.

    Uso:
        variable: es la variable que se quiere graficar
        titulo:   título del gráfico
        angulo:   opcional para los esfuerzos principales s1 y s2 y tmax
        nombre:   nombre del gráfico para guardarlo en formato .eps

    Para propósitos de graficación el EF se divide en 6 triángulos así: 

                           eta
                            ^
                            |
                            3
                            |\
                            | \
                            |  \
                            |(9)\
                            8----7
                            |(6)/|\
                            |  / | \
                            | /  |  \
                            |/(7)|(8)\
                            9----10---6
                            |\(2)|(3)/|\
                            | \  |  / | \
                            |  \ | /  |  \
                            |(1)\|/(4)|(5)\
                            1----4----5----2-----> xi

    El número entre paréntesis indica el número de cada EF triangular pequeño.                        
    '''
    
    nef = LaG.shape[0]    # número de elementos finitos en la malla original

    # se arma la matriz de correspondencia (LaG) de la nueva malla triangular
    LaG_t = np.zeros((9*nef, 3), dtype = int)
    for e in range(nef):
        LaG_t[9*e + 0, :] = LaG[e, [NL1,  NL4,  NL9 ]]
        LaG_t[9*e + 1, :] = LaG[e, [NL4,  NL10, NL9 ]]
        LaG_t[9*e + 2, :] = LaG[e, [NL4,  NL6,  NL10]]
        LaG_t[9*e + 3, :] = LaG[e, [NL4,  NL5,  NL6 ]]
        LaG_t[9*e + 4, :] = LaG[e, [NL5,  NL2,  NL6 ]]
        LaG_t[9*e + 5, :] = LaG[e, [NL9,  NL7,  NL8 ]]
        LaG_t[9*e + 6, :] = LaG[e, [NL9,  NL10, NL7 ]]
        LaG_t[9*e + 7, :] = LaG[e, [NL10, NL6,  NL7 ]]
        LaG_t[9*e + 8, :] = LaG[e, [NL8,  NL7,  NL3 ]]

    
    # se inicializa el lienzo
    fig, ax = plt.subplots(figsize=(7,7)) 

    # se encuentra el máximo en valor absoluto para ajustar el colorbar()
    val_max = np.max(np.abs(variable)) / 100     

    # se grafica la malla de EFS, los colores en cada triángulo y las curvas 
    # de nivel
    for e in range(nef):
        # se dibujan las aristas
        nod_ef = LaG[e, [NL1, NL4, NL5, NL2, NL6, NL7, NL3, NL8, NL9, NL1]]
        plt.plot(xnod[nod_ef, X], xnod[nod_ef, Y], lw = 0.5, color = 'gray')

    im = ax.tripcolor(xnod[:, X], xnod[:, Y], LaG_t, variable, cmap = 'bwr',
                      shading = 'gouraud', vmin = -val_max, vmax = val_max)
    ax.tricontour(xnod[:, X], xnod[:, Y], LaG_t, variable, 20)
    
    # a veces sale un warning, simplemente porque no existe la curva 0
    warnings.filterwarnings("ignore")
    ax.tricontour(xnod[:, X], xnod[:, Y], LaG_t, variable, levels=[0], linewidths=3)
    warnings.filterwarnings("default")

    fig.colorbar(im, ax = ax, format = '%6.3g')
                
    # para los esfuerzos principales se grafican las líneas que indiquen las
    # direcciones de los esfuerzos en cada nodo de la malla
    if angulo is not None:
       if type(angulo) is np.ndarray: 
           angulo = [ angulo ]
       for ang in angulo:
           ax.quiver(xnod[:, X], xnod[:, Y], 
                variable*np.cos(ang), variable*np.sin(ang), 
                headwidth=0, headlength=0, headaxislength=0, pivot='middle')
    
    # se especifican los ejes y el título, y se colocan los ejes iguales
    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')
    ax.set_title(titulo, fontsize=20)

    ax.set_aspect('equal')
    ax.autoscale(tight=True)
    plt.tight_layout()
    plt.savefig(nombre)
    plt.show()
