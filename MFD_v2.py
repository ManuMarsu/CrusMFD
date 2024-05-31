# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:18:33 2020

@author: manuel.collongues
Cerema / Laboratoire de Nancy / ERTD
"""

from __future__ import division
import numpy as np
from numba import cuda, njit, prange
import numba
import math
import datetime
from osgeo import gdal

# CUDA kernel
@cuda.jit
def my_kernel(dataMNT, directionsEcoulement, noDataValue):
    #sA = cuda.shared.array(shape=(tpb,tpb), dtype=float)
    #sB = cuda.shared.array(shape=(tpb,tpb), dtype=float)
    
    pos_x, pos_y = cuda.grid(2)
    if pos_x < dataMNT.shape[0] and pos_y < dataMNT.shape[1]:
        if dataMNT[pos_x, pos_y]  == noDataValue:
            dataMNT[pos_x, pos_y] = 99999
        dimx, dimy = dataMNT.shape
        voisinage = cuda.local.array(shape = 11,dtype = numba.int8) #0:centre /1-8:voisinage / 9:somme des écarts d'altitude / 10:nbr de cellules d'écoulement
        diff = cuda.local.array(shape = 11,dtype = numba.uint16)
        indice_min = cuda.local.array(shape = 1,dtype = numba.int8)
        min_voisins = cuda.local.array(shape = 1,dtype = numba.int16)
        indicat_dirEcoul = cuda.local.array(shape = 4,dtype = numba.int8) #0 : somme / 1: max /2: indice du max /3: indice de correction des arrondis /4: indice 2ème niveau de correction des arrondis
        
        
        #On gère les différents cas des bords et coins
        #voisinage[0:9] contient soit l'altitude des différents voisins immédiats du thread en cours, soit -1 si la donnée n'existe pas (voisinage[0] correspond au thread, [1] au nord-est, [2] au nord, ... - cf CalculReseauDrainage.pptx)
        if pos_x != 0 and pos_y != 0 and pos_x != dimx - 1 and pos_y != dimy - 1:       #cas général (intérieur)
            voisinage[0] = dataMNT[pos_x, pos_y]            #cellule centre
            voisinage[1] = dataMNT[pos_x + 1, pos_y - 1]    #nord-est
            voisinage[2] = dataMNT[pos_x, pos_y - 1]        #nord
            voisinage[3] = dataMNT[pos_x - 1, pos_y - 1]    #nord_ouest
            voisinage[4] = dataMNT[pos_x - 1, pos_y]        #ouest
            voisinage[5] = dataMNT[pos_x - 1, pos_y + 1]    #sud-ouest
            voisinage[6] = dataMNT[pos_x, pos_y + 1]        #sud
            voisinage[7] = dataMNT[pos_x + 1, pos_y + 1]    #sud-est
            voisinage[8] = dataMNT[pos_x + 1, pos_y]        #est
        if pos_x == 0 and pos_y != 0 and pos_y != dimy - 1:                             #cas x = 0 (hors coins)
            voisinage[0] = dataMNT[pos_x, pos_y]            #cellule centre
            voisinage[1] = dataMNT[pos_x + 1, pos_y - 1]    #nord-est
            voisinage[2] = dataMNT[pos_x, pos_y - 1]        #nord
            voisinage[3] = -1                               #nord_ouest
            voisinage[4] = -1                               #ouest
            voisinage[5] = -1                               #sud-ouest
            voisinage[6] = dataMNT[pos_x, pos_y + 1]        #sud
            voisinage[7] = dataMNT[pos_x + 1, pos_y + 1]    #sud-est
            voisinage[8] = dataMNT[pos_x + 1, pos_y]        #est
        if pos_x != 0 and pos_y == 0 and pos_x != dimx - 1:                             #cas y = 0 (hors coins)
            voisinage[0] = dataMNT[pos_x, pos_y]            #cellule centre
            voisinage[1] = -1                               #nord-est
            voisinage[2] = -1                               #nord
            voisinage[3] = -1                               #nord_ouest
            voisinage[4] = dataMNT[pos_x - 1, pos_y]        #ouest
            voisinage[5] = dataMNT[pos_x - 1, pos_y + 1]    #sud-ouest
            voisinage[6] = dataMNT[pos_x, pos_y + 1]        #sud
            voisinage[7] = dataMNT[pos_x + 1, pos_y + 1]    #sud-est
            voisinage[8] = dataMNT[pos_x + 1, pos_y]        #est
        if pos_y != 0 and pos_x == dimx - 1 and pos_y != dimy - 1:                      #cas x = max (hors coins)
            voisinage[0] = dataMNT[pos_x, pos_y]            #cellule centre
            voisinage[1] = -1                               #nord-est
            voisinage[2] = dataMNT[pos_x, pos_y - 1]        #nord
            voisinage[3] = dataMNT[pos_x - 1, pos_y - 1]    #nord_ouest
            voisinage[4] = dataMNT[pos_x - 1, pos_y]        #ouest
            voisinage[5] = dataMNT[pos_x - 1, pos_y + 1]    #sud-ouest
            voisinage[6] = dataMNT[pos_x, pos_y + 1]        #sud
            voisinage[7] = -1                               #sud-est
            voisinage[8] = -1                               #est
        if pos_x != 0 and pos_x != dimx - 1 and pos_y == dimy - 1:                      #cas y = max (hors coins)
            voisinage[0] = dataMNT[pos_x, pos_y]            #cellule centre
            voisinage[1] = dataMNT[pos_x + 1, pos_y - 1]    #nord-est
            voisinage[2] = dataMNT[pos_x, pos_y - 1]        #nord
            voisinage[3] = dataMNT[pos_x - 1, pos_y - 1]    #nord_ouest
            voisinage[4] = dataMNT[pos_x - 1, pos_y]        #ouest
            voisinage[5] = -1                               #sud-ouest
            voisinage[6] = -1                               #sud
            voisinage[7] = -1                               #sud-est
            voisinage[8] = dataMNT[pos_x + 1, pos_y]        #est
        if pos_x == 0 and pos_y == 0:                                                   #cas coin nord-ouest
            voisinage[0] = dataMNT[pos_x, pos_y]            #cellule centre
            voisinage[1] = -1                               #nord-est
            voisinage[2] = -1                               #nord
            voisinage[3] = -1                               #nord_ouest
            voisinage[4] = -1                               #ouest
            voisinage[5] = -1                               #sud-ouest
            voisinage[6] = dataMNT[pos_x, pos_y + 1]        #sud
            voisinage[7] = dataMNT[pos_x + 1, pos_y + 1]    #sud-est
            voisinage[8] = dataMNT[pos_x + 1, pos_y]        #est
        if pos_x == 0 and pos_y == dimy - 1:                                            #cas coin sud-ouest
            voisinage[0] = dataMNT[pos_x, pos_y]            #cellule centre
            voisinage[1] = dataMNT[pos_x + 1, pos_y - 1]    #nord-est
            voisinage[2] = dataMNT[pos_x, pos_y - 1]        #nord
            voisinage[3] = -1                               #nord_ouest
            voisinage[4] = -1                               #ouest
            voisinage[5] = -1                               #sud-ouest
            voisinage[6] = -1                               #sud
            voisinage[7] = -1                               #sud-est
            voisinage[8] = dataMNT[pos_x + 1, pos_y]        #est
        if pos_x == dimx - 1 and pos_y == 0:                                            #cas coin nord-est
            voisinage[0] = dataMNT[pos_x, pos_y]            #cellule centre
            voisinage[1] = -1                               #nord-est
            voisinage[2] = -1                               #nord
            voisinage[3] = -1                               #nord_ouest
            voisinage[4] = dataMNT[pos_x - 1, pos_y]        #ouest
            voisinage[5] = dataMNT[pos_x - 1, pos_y + 1]    #sud-ouest
            voisinage[6] = dataMNT[pos_x, pos_y + 1]        #sud
            voisinage[7] = -1                               #sud-est
            voisinage[8] = -1         
        if pos_x == dimx - 1 and pos_y == dimy - 1:                                     #cas coin sud-est
            voisinage[0] = dataMNT[pos_x, pos_y]            #cellule centre
            voisinage[1] = -1                               #nord-est
            voisinage[2] = dataMNT[pos_x, pos_y - 1]        #nord
            voisinage[3] = dataMNT[pos_x - 1, pos_y - 1]    #nord_ouest
            voisinage[4] = dataMNT[pos_x - 1, pos_y]        #ouest
            voisinage[5] = -1                               #sud-ouest
            voisinage[6] = -1                               #sud
            voisinage[7] = -1                               #sud-est
            voisinage[8] = -1                               #est
    
        indice_min[0] = 0
        min_voisins[0] = voisinage[0]
        diff[10] = 0
        #Ensuite on parcours (pour chaque thread) voisinage[0:9] et si l'altitude est supérieure, on remplace la valeur d'altitude par -2
        for i in range(9):
            if voisinage[i] > voisinage[0] and voisinage[i] > 0:
                voisinage[i] = -2
        for i in range(9):
            if voisinage[i] > 0:
                #On est alors dans le cas où voisinage[i] contient une altitude strictement inférieure à voisinage[0] (soit une case vers où on pourra avoir un écoulement)
                #Calcul de la différence d'altitude entre l'altitude du thread en cours et l'altitude du voisin étudié
                diff[i] = voisinage[0] - voisinage[i]
                diff[9] += diff[i]
                diff[10] += 1
        indicat_dirEcoul[1] = 0
        for i in range(9):
            if voisinage[i] > 0 and not (diff[9] == 0):
                directionsEcoulement[pos_x, pos_y,i] = valRel(diff[i], diff[9], diff[10], 1)
                indicat_dirEcoul[0] += valRel(diff[i], diff[9], diff[10], 1)
                if indicat_dirEcoul[1] < valRel(diff[i], diff[9], diff[10], 1):
                    indicat_dirEcoul[1] = valRel(diff[i], diff[9], diff[10], 1)
                    indicat_dirEcoul[2] = i
        
        #On corrige les répartitions d'écoulement dont la somme ne fait pas 100
        if indicat_dirEcoul[0] != 100:
            indicat_dirEcoul[3] = 100 - indicat_dirEcoul[0]
            for i in range(1,9):
                if directionsEcoulement[pos_x, pos_y, i] != 0 and indicat_dirEcoul[3] > 0:
                    directionsEcoulement[pos_x, pos_y, i] += 1
                    indicat_dirEcoul[3] -= 1
            if indicat_dirEcoul[3] > 0:
                for i in range(1,9):
                    if directionsEcoulement[pos_x, pos_y, i] != 0 and indicat_dirEcoul[3] > 0:
                        directionsEcoulement[pos_x, pos_y, i] += 1
                        indicat_dirEcoul[3] -= 1

                

@cuda.jit(device=True)
def valRel(diff_i, diff_tot, diff_nbr,conv):
    if conv == 0:
        res = 0
    elif conv == 1:
        res = (100 * diff_i) / diff_tot
    elif conv == 2:
        res = 0
    return res

#Fonction de test de la normalisation à 100 des directions d'écoulement de chaque pixel. 
@njit(parallel=True)
def test100(dirEc,somme):
    nbrAff = 0
    for i in prange(directionsEcoulement.shape[0]):
        for j in prange(directionsEcoulement.shape[1]):
            if not mnt[i,j] == 27108:
                res = 0
                for k in range(8):
                    res += directionsEcoulement[i,j,k+1]
                if res == somme and res != 0:
                    nbrAff += 1
    print(somme, " : ", nbrAff)


# Lecture du MNT
debut = datetime.datetime.now()
fichier_mnt = gdal.Open("Alti_Dpt10_AOC_Champagne.tif")
h_mnt = np.array(fichier_mnt.GetRasterBand(1).ReadAsArray())
(dim1,dim2) = h_mnt.shape
#print("\nMNT\ndim1 : ",dim1,"\ndim 2 : ",dim2)
print("...Lecture MNT terminée en : ", datetime.datetime.now() - debut)

d_mnt = cuda.to_device(np.ascontiguousarray(h_mnt * 100, dtype=np.uint16))
#mnt = cuda.to_device(np.ascontiguousarray(h_mnt[:limite,:limite], dtype = np.float32))
#mnt = np.array(h_mnt * 100, dtype=np.uint16)
#directionsEcoulement = cuda.to_device(np.zeros((dim1,dim2,9)))
#directionsEcoulement = np.zeros((dim1,dim2,9), dtype=np.uint8)
directionsEcoulement = cuda.device_array_like(np.zeros((dim1,dim2,9), dtype=np.uint8))

t0 = datetime.datetime.now()

# Host code   


# Appel du kernel GPU
tpb = 16,16 #threadsperblock
bpg_x = math.ceil(d_mnt.shape[0] / tpb[0]) #blockspergrid (-->9,7 si action sur la largeur 10 000 du raster)
bpg_y = math.ceil(d_mnt.shape[1] / tpb[1])
bpg = bpg_x, bpg_y
my_kernel[bpg, tpb](d_mnt, directionsEcoulement, 27108)
cuda.synchronize()
print("...Traitement GPU terminé en : ", datetime.datetime.now() - debut)

h_directionsEcoulement = directionsEcoulement.copy_to_host()

'''
for somme in range(90,111):
    test100(h_directionsEcoulement, somme)


for i in range(10):
    for j in range(2):
       print(mnt[i,j])
'''
# Récupération des résultats
t0 = datetime.datetime.now()
#directionEcoul = directionsEcoulement.copy_to_host()
print("...Récupération des résultats terminée en : ", datetime.datetime.now() - t0)







