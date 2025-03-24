#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
El propósito de este programa es leer la malla de EF creada en GMSH con el fin
de obtener los nodos y elementos finitos que pertenecen al borde externo de la
figura dada, y hallar la carga superficial a la que está sometido cada nodo de
cada uno de los EF del borde del sólido.

Al final del programa se exportan estos datos en un archivo de Excel que podrá
ser leído por el programa principal del taller.

Programa creado por: Alejandro Hincapié G.
"""

import numpy as np
import matplotlib.pyplot as plt
from leer_GMSH import LaG_from_msh
import pandas as pd


archivo = './malla/malla.msh'


# Se lee el archivo y se toman cada una de sus líneas:
m = open(archivo)
malla = m.readlines()
malla = [linea.rstrip('\n') for linea in malla]

# Se determina en qué lineas comienza y termina el reporte de nodos del
# archivo:
for i in range(len(malla)):
    if malla[i] == '$Nodes':
        inicio_nodos = i
    if malla[i] == '$EndNodes':
        fin_nodos = i

# Finalmente se comienza a leer el archivo para sacar los datos:

malla = malla[inicio_nodos+1:fin_nodos]

# Se leen los parámetros iniciales:
nblocks, nnodos = [int(n) for n in malla[0].split()[0:2]]

# Se inicializan las listas para cada una de las entidades que
# reporta el archivo:
puntos = []; nodos_puntos = []; xnod_puntos = []
bordes = []; nodos_bordes = []; xnod_bordes = []
superf = []; nodos_superf = []; xnod_superf = []


for j in range(1, len(malla)):
    line = malla[j]
    if len(line.split()) == 4 and line.split()[-1].isdigit(): # Se busca el reporte de cada bloque
        tipo_ent = int(line.split()[0])    # Punto, borde o superf.
        tag_ent  = int(line.split()[1])    # Identificador que usa gmsh

        nno_bloque = int(line.split()[-1]) # No. de nodos del bloque
        nodos_bloque = malla[j+1:j+1+nno_bloque]  # Lista de nodos del bloque
        nodos_bloque = [int(n) for n in nodos_bloque] # Se convierten a enteros

        xnod_b = malla[j+1+nno_bloque:j+1+2*nno_bloque]  # Coordenadas de nodos

        # Se reportan las coordenadas como una "matriz":
        xnod_bloque = []
        for l in xnod_b:
            coord = [float(n) for n in l.split()]
            xnod_bloque.append(coord)

        # Finalmente se agregan los datos leídos a la lista corres-
        # pondiente según la entidad (punto, línea o superficie):
        if tipo_ent == 0:
            puntos.append(str(tag_ent))
            nodos_puntos.append(nodos_bloque)
            xnod_puntos.append(xnod_bloque)
        elif tipo_ent == 1:
            bordes.append(str(tag_ent))
            nodos_bordes.append(nodos_bloque)
            xnod_bordes.append(xnod_bloque)
        elif tipo_ent == 2:
            superf.append(str(tag_ent))
            nodos_superf.append(nodos_bloque)
            xnod_superf.append(xnod_bloque)

# %% Se ensambla finalmente la matriz xnod completa:

xnod = np.empty((nnodos, 3))

# Primero se adicionan los nodos correspondientes a los puntos:
for i in range(len(puntos)):
    idx = np.array(nodos_puntos[i]) - 1
    xnod[idx, :] = np.array(xnod_puntos[i])

# Luego se agregan los nodos correspondientes a los bordes:
for i in range(len(bordes)):
    idx = np.array(nodos_bordes[i]) - 1
    xnod[idx, :] = np.array(xnod_bordes[i])

# Finalmente se agregan los nodos correspondientes a la superficie:
for i in range(len(superf)):
    idx = np.array(nodos_superf[i]) - 1
    xnod[idx, :] = np.array(xnod_superf[i])

# Se toman únicamente las dos primeras columnas de xnod (coordenadas (x, y)):
xnod = xnod[:, 0:2]

# %% Se determinan cuáles curvas y puntos están contenidos en el borde cargado:

puntos_ex = [3, 4, 5, 9]  # Puntos físicos en el borde cargado
bordes_ex = [5, 6, 7]     # Bordes físicos en el borde cargado

idx = []  # contiene todos los nodos del borde externo
for p in puntos_ex:
    idx.extend(nodos_puntos[p-1])
for b in bordes_ex:
    idx.extend(nodos_bordes[b-1])

idx = np.array(idx) - 1

LaG = LaG_from_msh(archivo)  # Se obtiene la matriz LaG


# %% Se obtienen los elementos finitos que limitan con el borde exterior

EF   = []  # Lista de EFs del borde externo
lado = []  # Lado en que está aplicada la carga

for i in range(LaG.shape[0]):
    fila = LaG[i]
    # Se buscan los EF en los cuales 4 elementos pertenecen al borde exterior:
    if fila[np.isin(fila, idx)].size == 4:
        EF.append(i)

        # Y se halla el lado que limita con el borde exterior:
        if np.all(np.isin(fila[[0, 3, 4, 1]], idx)):
            lado.append('1452')
        elif np.all(np.isin(fila[[1, 5, 6, 2]], idx)):
            lado.append('2673')
        elif np.all(np.isin(fila[[2, 7, 8, 0]], idx)):
            lado.append('3891')
        else:
            raise ValueError('Hay un problema con los lados')


# %% Se almacenan los datos en un DataFrame de Pandas:

df = pd.DataFrame(columns=['elemento', 'lado', 'tix', 'tiy', 'tjx', 'tjy',
                           'tkx', 'tky', 'tlx', 'tly'])

df['elemento'] = np.array(EF) + 1
df['lado'] = lado

# %% Finalmente, se calcula la carga superficial (en KPa) a la que está someti-
# do cada nodo de los EF:

P = 50  # Carga en kN/m
t = 0.2  # Espesor en metros


for i in range(len(EF)):
    if   lado[i] == '1452': nod = LaG[EF[i]][[0,3,4,1]]
    elif lado[i] == '2673': nod = LaG[EF[i]][[1,5,6,2]]
    elif lado[i] == '3891': nod = LaG[EF[i]][[2,7,8,0]]
    else: raise ValueError('Hay un error en los lados del EF')

    x = xnod[nod, 0]
    y = xnod[nod, 1]
    ang = np.arctan2(y, x)

    fx = 0
    fy = -P/t

    # Se reemplazan las fuerzas en el DataFrame:
    df.iloc[i, [2, 4, 6, 8]] = fx
    df.iloc[i, [3, 5, 7, 9]] = fy

# %% Se exportan los datos a Excel:

df.to_excel('./malla/superficiales.xlsx', sheet_name='carga_distr',
            index=False)

# %% Se grafican los EF del borde externo (opcional):

nef = LaG.shape[0]
nno = xnod.shape[0]
plotear = True

if plotear:
    cg = np.zeros((nef,2))  # almacena el centro de gravedad de los EF

    plt.figure(figsize=(6,6))
    for e in EF:
       nod_ef = LaG[e, [0,3,4,1,5,6,2,7,8,0]]
       plt.plot(xnod[nod_ef, 0], xnod[nod_ef, 1], 'b')

       # se calcula la posición del centro de gravedad del triángulo
       cg[e] = np.mean(xnod[LaG[e,:], :], axis=0)

       # y se reporta el número del elemento actual
       plt.text(cg[e,0], cg[e,1], f'{e+1}', horizontalalignment='center',
                                            verticalalignment='center',  color='b')

    plt.plot(xnod[idx,0], xnod[idx,1], 'r.')
    for i in idx:
       plt.text(xnod[i,0], xnod[i,1], f'{i+1}', color='r')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.title('Malla de elementos finitos')
    plt.show()
