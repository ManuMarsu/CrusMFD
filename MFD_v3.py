# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:18:33 2020

@author: manuel.collongues
Cerema / Laboratoire de Nancy / ERTD
"""

from __future__ import division
import numpy as np
from numba import cuda, njit, prange
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numba
import math
import datetime
from osgeo import gdal

@cuda.jit
def myk_bruitageMNT(mnt_source, mnt_bruite, noDataValue, amplitudeBruit, rng_states_x, rng_states_y, bruitageActif):
    pos_x, pos_y = cuda.grid(2)
    x = xoroshiro128p_uniform_float32(rng_states_x, pos_x)
    y = xoroshiro128p_uniform_float32(rng_states_y, pos_y)
    bruitage = int(amplitudeBruit * (x + y) - amplitudeBruit)
    if mnt_source[pos_x, pos_y] == noDataValue:
        mnt_bruite[pos_x, pos_y] = noDataValue
    else:
        mnt_bruite[pos_x, pos_y] = mnt_source[pos_x, pos_y] + bruitageActif * bruitage

@cuda.jit
def myk_comblementDepressions(dataMNT, mnt_filled, mnt_codes_depr, noDataValue, mnt_ind_depr, mnt_ind_depr_tot, mnt_alt_cretes, celluleTraitees):
    pos_x, pos_y = cuda.grid(2)
    dimx, dimy = dataMNT.shape
    celluleTraitees[pos_x, pos_y] = False
    
    voisinage = cuda.local.array(shape = 10,dtype = numba.int32) #0:centre /1-8:voisinage / 9:compte du nbr de cellules voisines inférieures au centre
    voisinage_tourne = cuda.local.array(shape = 10,dtype = numba.int32)

    mnt_filled[pos_x, pos_y] = dataMNT[pos_x, pos_y]
    mnt_ind_depr[pos_x, pos_y] = 0
    mnt_ind_depr_tot[pos_x, pos_y] = 0
    
    #On gère les différents cas des bords et coins
    voisinage = voisins(dataMNT, pos_x, pos_y, dimx, dimy, voisinage)
    
    if voisinage[9] == 0:
        altExut = exutoiresVoisins(voisinage)
        mnt_filled[pos_x, pos_y] = altExut
     
    #Visualisation de la répartition du nombre de cellules d'écoulement
    if dataMNT[pos_x, pos_y]  == noDataValue:
        mnt_filled[pos_x, pos_y] = noDataValue
        mnt_ind_depr[pos_x, pos_y] = 0
        celluleTraitees[pos_x, pos_y] = True
    elif dataMNT[pos_x, pos_y]  != noDataValue: #¨On élimine les valeurs NoData et bords
        for j in range(10):
            mnt_ind_depr[pos_x, pos_y] = 0
            if voisinage[9] == j:
                mnt_codes_depr[pos_x, pos_y] = j
                mnt_ind_depr[pos_x, pos_y] = 1
                
            if j == 9:
                mnt_ind_depr[pos_x, pos_y] = 1
        
            cuda.atomic.add(mnt_ind_depr_tot, (0, j), mnt_ind_depr[pos_x, pos_y])
    
    #Visualisation des points de crête
    mnt_alt_cretes[pos_x, pos_y] = crete(voisinage, voisinage_tourne)
    
@cuda.jit
def myk_copieMNT(mnt_source, mnt_dest):
    pos_x, pos_y = cuda.grid(2)
    mnt_dest[pos_x, pos_y] = mnt_source[pos_x, pos_y]
        
@cuda.jit
def myk_directionsEcoulement(dataMNT, directionsEcoulement, noDataValue):
    
    pos_x, pos_y = cuda.grid(2)
    if pos_x < dataMNT.shape[0] and pos_y < dataMNT.shape[1]:
        if dataMNT[pos_x, pos_y]  == noDataValue:
            dataMNT[pos_x, pos_y] = 99999
        dimx, dimy = dataMNT.shape
        voisinage = cuda.local.array(shape = 11,dtype = numba.int32) #0:centre /1-8:voisinage / 9:somme des écarts d'altitude / 10:nbr de cellules d'écoulement
        diff = cuda.local.array(shape = 11,dtype = numba.uint16)
        indice_min = cuda.local.array(shape = 1,dtype = numba.int32)
        min_voisins = cuda.local.array(shape = 1,dtype = numba.int16)
        indicat_dirEcoul = cuda.local.array(shape = 4,dtype = numba.int32) #0 : somme / 1: max /2: indice du max /3: indice de correction des arrondis /4: indice 2ème niveau de correction des arrondis
        
        
        #On gère les différents cas des bords et coins
        #voisinage[0:9] contient soit l'altitude des différents voisins immédiats du thread en cours, soit -1 si la donnée n'existe pas (voisinage[0] correspond au thread, [1] au nord-est, [2] au nord, ... - cf CalculReseauDrainage.pptx)
        voisinage = voisins(dataMNT, pos_x, pos_y, dimx, dimy, voisinage)

    
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

@cuda.jit
def myk_cellulesDrainees(dataMNT, directionsEcoulement, noDataValue, cellDrainees, cellTraitee):
    
    pos_x, pos_y = cuda.grid(2)
    if pos_x < dataMNT.shape[0] and pos_y < dataMNT.shape[1]:
        if dataMNT[pos_x, pos_y]  == noDataValue:
            cellDrainees[pos_x, pos_y] = 0
        else:
            dimx, dimy = dataMNT.shape
            locIndice = cuda.local.array(shape = (10, 3),dtype = numba.int32) 
            
            for i in range(1,9):
                if i == 3 or i == 4 or i == 5:
                    locIndice[i,0] = -1
                elif i == 2 or i == 6:
                    locIndice[i,0] = 0
                else:
                    locIndice[i,0] = 1
                
                if i == 1 or i == 2 or i == 3:
                    locIndice[i,1] = -1
                elif i == 4 or i == 8:
                    locIndice[i,1] = 0
                else:
                    locIndice[i,1] = 1
                    
                if i <= 4:
                    locIndice[i,2] = i + 4
                else:
                    locIndice[i,2] = i - 4
            
            test = test_amav(pos_x, pos_y, dimx, dimy, cellTraitee, locIndice, directionsEcoulement)
            if test:
                aireDraineeCelluleActive = 0
                for i in range(1,9):
                    x = pos_x + locIndice[i,0]
                    y = pos_y + locIndice[i,1]
                    indice = locIndice[i,2]
                    if x < dimx and y < dimy and x > 0 and y > 0:
                        if directionsEcoulement[x, y, indice] > 0:
                            aireDraineeCelluleActive += directionsEcoulement[x, y, indice] * 100
                cellDrainees[pos_x, pos_y] = 1 + aireDraineeCelluleActive
                cellTraitee[pos_x, pos_y] = True
    #cuda.atomic.compare_and_swap(cellTraitee, )
        #cellDrainees[pos_x, pos_y] = aireDr(pos_x, pos_y, dimx, dimy, dataMNT, directionsEcoulement, locIndice)
        
@cuda.jit(device=True)
def test_amav(pos_x, pos_y, dimx, dimy, cellTraitee, locIndice, directionsEcoulement):
    result = True
    for i in range(1,9):
        x = pos_x + locIndice[i,0]
        y = pos_y + locIndice[i,1]
        indice = locIndice[i,2]
        if x < dimx and y < dimy and x > 0 and y > 0:
            result = result and (cellTraitee[x, y] == True or directionsEcoulement[x, y, indice] == 0)
    return result

@cuda.jit(device=True)
def aireDr(pos_x, pos_y, dimx, dimy, dataMNT, directionsEcoulement, locIndice):
    aireDraineeCelluleActive = 1
    for i in range(1,9):
        x = pos_x + locIndice[i,0]
        y = pos_y + locIndice[i,1]
        indice = locIndice[i,2]
        if x < dimx and y < dimy and x > 0 and y > 0:
            if directionsEcoulement[x, y, indice] > 0:
                aireDraineeCelluleActive += directionsEcoulement[x, y, indice] * 1
                aire2 = aireDr(x, y, dimx, dimy, dataMNT, directionsEcoulement, locIndice) / 100
                #aireDraineeCelluleActive += (directionsEcoulement[x, y, indice] * aireDr(x, y, dimx, dimy, dataMNT, directionsEcoulement, locIndice)) / 100
    
    return aireDraineeCelluleActive
    

@cuda.jit(device=True)
def tourne_voisins(voisinage, voisinage_tourne, agl): #agl représente le nombre de quart de tour de rotation dans le sens des aiguilles d'une montre
    if agl == 1:
        voisinage_tourne[0] = voisinage[0]
        voisinage_tourne[1] = voisinage[3]
        voisinage_tourne[2] = voisinage[4]
        voisinage_tourne[3] = voisinage[5]
        voisinage_tourne[4] = voisinage[6]
        voisinage_tourne[5] = voisinage[7]
        voisinage_tourne[6] = voisinage[8]
        voisinage_tourne[7] = voisinage[1]
        voisinage_tourne[8] = voisinage[2]
    elif agl == 2:
        voisinage_tourne[0] = voisinage[0]
        voisinage_tourne[1] = voisinage[5]
        voisinage_tourne[2] = voisinage[6]
        voisinage_tourne[3] = voisinage[7]
        voisinage_tourne[4] = voisinage[8]
        voisinage_tourne[5] = voisinage[1]
        voisinage_tourne[6] = voisinage[2]
        voisinage_tourne[7] = voisinage[3]
        voisinage_tourne[8] = voisinage[4]
    elif agl == 3:
        voisinage_tourne[0] = voisinage[0]
        voisinage_tourne[1] = voisinage[7]
        voisinage_tourne[2] = voisinage[8]
        voisinage_tourne[3] = voisinage[1]
        voisinage_tourne[4] = voisinage[2]
        voisinage_tourne[5] = voisinage[3]
        voisinage_tourne[6] = voisinage[4]
        voisinage_tourne[7] = voisinage[5]
        voisinage_tourne[8] = voisinage[6]    
    else:
        voisinage_tourne[0] = voisinage[0]
        voisinage_tourne[1] = voisinage[1]
        voisinage_tourne[2] = voisinage[2]
        voisinage_tourne[3] = voisinage[3]
        voisinage_tourne[4] = voisinage[4]
        voisinage_tourne[5] = voisinage[5]
        voisinage_tourne[6] = voisinage[6]
        voisinage_tourne[7] = voisinage[7]
        voisinage_tourne[8] = voisinage[8]
    return voisinage_tourne
    
@cuda.jit(device=True)
def crete(voisinage, voisinage_tourne):
    alt_crete = 0
    # cas croix
    c1 = voisinage[1] < voisinage[0] and voisinage[1] < voisinage[2] and voisinage[1] < voisinage[8]
    c7 = voisinage[7] < voisinage[0] and voisinage[7] < voisinage[6] and voisinage[7] < voisinage[8]
    c5 = voisinage[5] < voisinage[0] and voisinage[5] < voisinage[6] and voisinage[5] < voisinage[4]
    c3 = voisinage[3] < voisinage[0] and voisinage[3] < voisinage[2] and voisinage[3] < voisinage[4]
    cCroix = c1 and c7 and c5 and c3
    # cas "t"     
    cT = False
    for i in range(4):
        voisinage_tourne = tourne_voisins(voisinage, voisinage_tourne, i)
        c1 = voisinage_tourne[1] < voisinage_tourne[2]
        c7 = voisinage_tourne[7] < voisinage_tourne[6]
        c5 = voisinage_tourne[5] < voisinage_tourne[0] and voisinage_tourne[5] < voisinage_tourne[6] and voisinage_tourne[5] < voisinage_tourne[4]
        c3 = voisinage_tourne[3] < voisinage_tourne[0] and voisinage_tourne[3] < voisinage_tourne[2] and voisinage_tourne[3] < voisinage_tourne[4]
        c8 = voisinage_tourne[8] < voisinage_tourne[0]
        cT = cT or (c1 and c3 and c5 and c7 and c8)
    # cas "vertical" + "horizontal"
    cVH = False
    for i in range(2):
        voisinage_tourne = tourne_voisins(voisinage, voisinage_tourne, i)
        c1 = voisinage_tourne[1] < voisinage_tourne[2]
        c7 = voisinage_tourne[7] < voisinage_tourne[6]
        c5 = voisinage_tourne[5] < voisinage_tourne[6]
        c3 = voisinage_tourne[3] < voisinage_tourne[2]
        c8 = voisinage_tourne[8] < voisinage_tourne[0]
        c4 = voisinage_tourne[4] < voisinage_tourne[0]
        cVH = cVH or (c1 and c3 and c5 and c7 and c8 and c4)
    # cas "coin"
    cCoin = False
    for i in range(4):
        voisinage_tourne = tourne_voisins(voisinage, voisinage_tourne, i)
        c1 = voisinage_tourne[1] < voisinage_tourne[2]
        c5 = voisinage_tourne[5] < voisinage_tourne[4]
        c3 = voisinage_tourne[3] < voisinage_tourne[0] and voisinage_tourne[3] < voisinage_tourne[2] and voisinage_tourne[3] < voisinage_tourne[4]
        c8 = voisinage_tourne[8] < voisinage_tourne[0]
        c6 = voisinage_tourne[6] < voisinage_tourne[0]
        cCoin = cCoin or (c1 and c5 and c3 and c8 and c6)
    # Conclusion
    if cCroix or cT or cVH or cCoin:
        alt_crete = voisinage[0]
    return alt_crete

@cuda.jit(device=True)
def exutoiresVoisins(voisinage):
    mini1 = 0
    indice_mini1 = 0
    mini2 = 0
    for i in range(1,9):
        if voisinage[i] > mini1:
            mini1 = voisinage[i]
            indice_mini1 = i
    for i in range(1,9):
        if voisinage[i] > mini2:
            if i != indice_mini1:
                mini1 = voisinage[i]
                indice_mini1 = i
    altExut = int((mini1 + mini2) / 2) + 1
    return altExut

@cuda.jit(device=True)
def voisins(dataMNT, pos_x, pos_y, dimx, dimy, voisinage):
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
    if pos_x == dimx - 1 and pos_y == dimy - 1: 
        voisinage[0] = dataMNT[pos_x, pos_y]            #cellule centre
        voisinage[1] = -1                               #nord-est
        voisinage[2] = dataMNT[pos_x, pos_y - 1]        #nord
        voisinage[3] = dataMNT[pos_x - 1, pos_y - 1]    #nord_ouest
        voisinage[4] = dataMNT[pos_x - 1, pos_y]        #ouest
        voisinage[5] = -1                               #sud-ouest
        voisinage[6] = -1                               #sud
        voisinage[7] = -1                               #sud-est
        voisinage[8] = -1                               #est
    voisinage[9] = 0
    for i in range(1,9):
        if voisinage[i] > 0 and voisinage[i] < voisinage[0]:
            voisinage[9] += 1
    return voisinage

@cuda.jit(device=True)
def valRel(diff_i, diff_tot, diff_nbr,conv):
    if conv == 0:
        res = 0
    elif conv == 1:
        res = (100 * diff_i) / diff_tot
    elif conv == 2:
        res = 0
    return res


# Lecture du MNT
debut = datetime.datetime.now()
fichier_mnt = gdal.Open("Alti_Dpt10_AOC_Champagne.tif")
h_mnt = np.array(fichier_mnt.GetRasterBand(1).ReadAsArray())
NDV = fichier_mnt.GetRasterBand(1).GetNoDataValue()
#print("NDV : ", NDV)
#print(h_mnt[0:1,0:5])
(dim1,dim2) = h_mnt.shape
#print("...Lecture MNT terminée en : ", datetime.datetime.now() - debut)

# Transfert des données MNT sur le device (on transforme les valeurs flottantes en mètres en valeurs entières en cm)
d_mnt = cuda.to_device(np.ascontiguousarray(h_mnt * 100, dtype=np.uint32))
# Allocation des variables sur le device
d_directionsEcoulement = cuda.device_array_like(np.zeros((dim1,dim2,9), dtype=np.uint8))
d_mnt_filled = cuda.device_array_like(d_mnt)
d_mnt_codes_depr = cuda.device_array_like(d_mnt)
d_mnt_bruite = cuda.device_array_like(d_mnt)
d_mnt_ind_depr = cuda.device_array_like(d_mnt)
d_mnt_ind_depr_tot = cuda.device_array_like(d_mnt)
d_mnt_alt_cretes = cuda.device_array_like(d_mnt)
d_cellDrainees = cuda.device_array_like(d_mnt)
d_cellTraitee = cuda.device_array_like(np.zeros((dim1,dim2), dtype=np.bool_))

test_mnt = d_mnt[0:1,0:5].copy_to_host()
print("NoDataValue sur device : ", test_mnt[0,0])


nbrIterations = 1
amplitudeBruit = 20
bruitageActif = 1

stats = np.zeros((nbrIterations,10), dtype=np.int64)


t1 = datetime.datetime.now()

tpb = 16,16 #threadsperblock
bpg_x = math.ceil(d_mnt.shape[0] / tpb[0]) #blockspergrid (-->9,7 si action sur la largeur 10 000 du raster)
bpg_y = math.ceil(d_mnt.shape[1] / tpb[1])
rng_states_x = create_xoroshiro128p_states(tpb[0] * bpg_x, seed=1)
rng_states_y = create_xoroshiro128p_states(tpb[1] * bpg_y, seed=1)
bpg = bpg_x, bpg_y

for iterationsBruitage in range(nbrIterations):
    # [GPU] - bruitage du MNT
    t0 = datetime.datetime.now()
    myk_bruitageMNT[bpg, tpb](d_mnt, d_mnt_bruite, 4284967396, amplitudeBruit, rng_states_x, rng_states_y, bruitageActif)
    cuda.synchronize()
    #print("...[GPU] - Comblement des zones dépressionnaires terminé en : ", datetime.datetime.now() - t0)
    
    
    
    # [GPU] - Calcul des zones dépressionnaires
    t0 = datetime.datetime.now()
    myk_comblementDepressions[bpg, tpb](d_mnt_bruite, d_mnt_filled, d_mnt_codes_depr, 4284967396, d_mnt_ind_depr, d_mnt_ind_depr_tot, d_mnt_alt_cretes, d_cellTraitee)
    cuda.synchronize()
    #print("...[GPU] - Comblement des zones dépressionnaires terminé en : ", datetime.datetime.now() - t0)
    
    h_mnt_ind_depr_tot = d_mnt_ind_depr_tot[0:1,0:10].copy_to_host()
    #print("Nombre de cellules d'écoulement : ", 9, " : ", h_mnt_ind_depr_tot[0, 9])
    for i in range(0,1):
        stats[iterationsBruitage, i] = h_mnt_ind_depr_tot[0, i]
        #print("Nombre de cellules d'écoulement : ", i, " : ", h_mnt_ind_depr_tot[0, i])
    
    for k in range(20):
        if h_mnt_ind_depr_tot[0,0] > 0:
            myk_copieMNT[bpg, tpb](d_mnt_filled, d_mnt_bruite) #copie filled vers bruite
            cuda.synchronize()
            
            # [GPU] - Calcul des zones dépressionnaires
            t0 = datetime.datetime.now()
            myk_comblementDepressions[bpg, tpb](d_mnt_bruite, d_mnt_filled, d_mnt_codes_depr, 4284967396, d_mnt_ind_depr, d_mnt_ind_depr_tot, d_mnt_alt_cretes, d_cellTraitee)
            cuda.synchronize()
            #print("...[GPU] - Comblement des zones dépressionnaires terminé en : ", datetime.datetime.now() - t0)
    
            h_mnt_ind_depr_tot = d_mnt_ind_depr_tot[0:1,0:10].copy_to_host()
            
            #print("Nombre de cellules d'écoulement : ", 9, " : ", h_mnt_ind_depr_tot[0, 9])
            for i in range(0,1):
                stats[iterationsBruitage, i] = h_mnt_ind_depr_tot[0, i]
                if k == 19:
                    print("Nombre de cellules d'écoulement : ", i, " : ", h_mnt_ind_depr_tot[0, i])
            
    
    
    
    # [GPU] - Calcul des directions d'écoulement
    t0 = datetime.datetime.now()
    myk_directionsEcoulement[bpg, tpb](d_mnt_bruite, d_directionsEcoulement, 27108)
    cuda.synchronize()
    #print("...[GPU] - Calcul des directions d'écoulement terminé en : ", datetime.datetime.now() - t0)
    
    # [GPU] - Calcul du nombre de cellules drainées
    t0 = datetime.datetime.now()
    myk_cellulesDrainees[bpg, tpb](d_mnt_bruite, d_directionsEcoulement, 4284967396, d_cellDrainees, d_cellTraitee)
    cuda.synchronize()
    print("...[GPU] - Calcul des cellules drainées terminé en : ", datetime.datetime.now() - t0)
    
    
    
print("...[GPU] - Boucle complète de", nbrIterations, "itérations terminée en :", datetime.datetime.now() - t1)

'''
for i in range(10):
    print("Nombre de cellules d'écoulement : ", i, " : ", int(stats[:, i].mean()))

h_directionsEcoulement = d_directionsEcoulement.copy_to_host()    

'''
'''
# Ecriture du fichier de sortie
h_mnt_filled = mnt_codes_depr.copy_to_host()

debut = datetime.datetime.now()
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create("mnt_filled_4.tif", dim2, dim1, 1, gdal.GDT_Int32)
outdata.SetGeoTransform(fichier_mnt.GetGeoTransform())##sets same geotransform as input
outdata.SetProjection(fichier_mnt.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(h_mnt_filled)
outdata.GetRasterBand(1).SetNoDataValue(99999)##if you want these values transparent
outdata.FlushCache() ##saves to disk!!
print("...Ecriture fichier sortie terminée en : ", datetime.datetime.now() - debut)
'''


# Ecriture du fichier de sortie
h_mnt_alt_cretes = d_mnt_alt_cretes.copy_to_host()

debut = datetime.datetime.now()
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create("mnt_lignesCretes4.tif", dim2, dim1, 1, gdal.GDT_Int32)
outdata.SetGeoTransform(fichier_mnt.GetGeoTransform())##sets same geotransform as input
outdata.SetProjection(fichier_mnt.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(h_mnt_alt_cretes)
outdata.GetRasterBand(1).SetNoDataValue(99999)##if you want these values transparent
outdata.FlushCache() ##saves to disk!!
print("...Ecriture fichier sortie terminée en : ", datetime.datetime.now() - debut)



'''
# Récupération des résultats
t0 = datetime.datetime.now()
#directionEcoul = directionsEcoulement.copy_to_host()
print("...Récupération des résultats terminée en : ", datetime.datetime.now() - t0)
'''






