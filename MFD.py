# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:18:33 2020

@author: manuel.collongues
Cerema / Laboratoire de Nancy / ERTD
"""

from __future__ import division
import numpy as np
from numba import cuda
import numba
import math
import datetime
from osgeo import gdal
import xlsxwriter

# CUDA kernel
@cuda.jit
def my_kernel(dataMNT, dirEcoul):
    #sA = cuda.shared.array(shape=(tpb,tpb), dtype=float)
    #sB = cuda.shared.array(shape=(tpb,tpb), dtype=float)
    
    pos_x, pos_y = cuda.grid(2)
        
    dimx, dimy = dataMNT.shape
    voisinage = cuda.local.array(shape = 9,dtype = numba.int8)
    indice_min = cuda.local.array(shape = 1,dtype = numba.int8)
    min_voisins = cuda.local.array(shape = 1,dtype = numba.int16)
    
    #On gère les différents cas des bords et coins
    #voisinage[0:9] contient soit l'altitude des différents voisins immédiats du thread en cours, soit -1 si la donnée n'existe pas
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
    #Ensuite on parcours (pour chaque thread) voisinage[0:9] et si l'altitude est supérieure
    for i in range(len(voisinage)):
        if voisinage[i] < min_voisins[0] and voisinage[i] > 0:
            min_voisins[0] = voisinage[i]
            indice_min[0] = i
    dirEcoul[pos_x, pos_y] = indice_min[0]
    
    cuda.syncthreads()




# Lecture du MNT
debut = datetime.datetime.now()
fichier_mnt = gdal.Open("Alti_Dpt10_AOC_Champagne.tif")
h_mnt = np.array(fichier_mnt.GetRasterBand(1).ReadAsArray())
(dim1,dim2) = h_mnt.shape
#print("\nMNT\ndim1 : ",dim1,"\ndim 2 : ",dim2)
print("...Lecture MNT terminée en : ", datetime.datetime.now() - debut)

mnt = cuda.to_device(np.ascontiguousarray(h_mnt * 100, dtype = np.int))
directionsEcoulement = cuda.to_device(np.ascontiguousarray(np.zeros(h_mnt.shape), dtype = np.int))

t0 = datetime.datetime.now()



# Host code   


# Appel du kernel GPU
tpb = 32,32 #threadsperblock
bpg_x = math.ceil(mnt.shape[0] / tpb[0]) #blockspergrid (-->9,7 si action sur la largeur 10 000 du raster)
bpg_y = math.ceil(mnt.shape[1] / tpb[1])
bpg = bpg_x, bpg_y
my_kernel[bpg, tpb](mnt, directionsEcoulement)
print("...Traitement GPU terminé en : ", datetime.datetime.now() - debut)












