#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 17:19:38 2020

@author: alejandro
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


archivo = './malla/malla.msh'
#X, Y = 0, 1
#NL1, NL2, NL3, NL4, NL5, NL6, NL7, NL8, NL9, NL10 = range(10)
#g = 9.81  # [m/s²]   aceleración de la gravedad


def xnod_from_msh(archivo):
    ''' Obtiene la matriz de coordenadas de los nodos que contiene la malla de
        EF a trabajar, a partir del archivo .msh exportado por el programa GMSH.
    '''

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
        if len(line.split()) == 4:             # Se busca el reporte de cada bloque
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

    return xnod

def LaG_from_msh(archivo):
    ''' Obtiene la matriz de interconexión nodal (LaG) que contiene la malla de
        EF a trabajar, a partir del archivo .msh exportado por el programa GMSH.
    '''
    # Se lee el archivo y se toman cada una de sus líneas:
    m = open(archivo)
    malla = m.readlines()
    malla = [linea.rstrip('\n') for linea in malla]

    # Se determina en qué lineas comienza y termina el reporte de nodos del
    # archivo:
    for i in range(len(malla)):
        if malla[i] == '$Elements':
            inicio_elem = i
        if malla[i] == '$EndElements':
            fin_elem = i

    malla = malla[inicio_elem:fin_elem]  # Se toman solo las líneas necesarias

    p = malla[1]  # Parámetros a leer
    nblocks, nelem = [int(n) for n in p.split()[0:2]]

    for i in range(1, len(malla)):
        linea = [int(n) for n in malla[i].split()]
        if len(linea) == 4 and linea[0] == 2:
            nel = linea[-1]  # Número de EF triangulares
            LaG = []
            for j in range(i+1, i+1+nel):
                line = [int(n) for n in malla[j].split()]
                LaG.append(line)
    LaG = np.array(LaG)
    LaG = LaG[:, 1:]-1

    return LaG

