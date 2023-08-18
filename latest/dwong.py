import numpy as np
import awkward as ak
import os,sys
import math
from sklearn.cluster import KMeans
from scipy import spatial
import scipy.stats as sp
from sklearn.cluster import Birch
import uproot3 as uproot



"""

import time
import mplhep as hep
import matplotlib.pyplot as plt
plt.style.use(hep.style.ROOT)
import uproot as uproot4
import numba
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import rcParams
import pandas as pd
from scipy.stats import halfnorm
import random
from scipy.stats import binned_statistic
from scipy.stats import binned_statistic_2d
from scipy.optimize import curve_fit
from scipy.stats import crystalball
from lmfit import Model
import copy
from sklearn.mixture import GaussianMixture#clustering
from sklearn.cluster import DBSCAN#clustering
from scipy.stats import binned_statistic
from scipy.stats import halfnorm
from shapely.geometry import Point, Polygon





"""
#--------------------------------------------------------------------------------------------------------------------------------------------
evt_1000 = {
  "electron": "electron_520_1000eve.root",
  "pi+": "pi+_520_1000eve.root",
  "pi-": "pi-_520_1000eve.root",
  "pi0": "pi0_520_1000eve.root",
  "positron": "positron_520_1000eve.root",
  "muon": "muon_520_1000eve.root",
  "photon": "gamma_520_1000eve.root",
  "klong": "klong_520_1000eve.root"
}
#--------------------------------------------------------------------------------------------------------------------------------------------
def getData(fname="", procName="Events"):
    kn_dict = uproot.open(fname)["Events"].arrays()
    kn_dict_ak1 = {name.decode(): ak.from_awkward0(array) for name, array in kn_dict.items()}
    kn_events = ak.zip({"Electrons":ak.zip({
                                            "ge":      kn_dict_ak1["ge"],
                                            "gvx":      kn_dict_ak1["gvx"],
                                            "gvy":      kn_dict_ak1["gvy"],
                                            "gvz":      kn_dict_ak1["gvz"],
                                            "gpx":      kn_dict_ak1["gpx"],
                                            "gpy":      kn_dict_ak1["gpy"],
                                            "gpz":      kn_dict_ak1["gpz"],
                                        }),
                        "Hits":ak.zip({
                                      "detID":   kn_dict_ak1["hit_detID"],
                                      "edep":    kn_dict_ak1["hit_edep"],
                                      "elmID":   kn_dict_ak1["hit_elmID"],
                                      "truthx":  kn_dict_ak1["hit_truthx"],
                                      "truthy":  kn_dict_ak1["hit_truthy"],
                                      "truthz":  kn_dict_ak1["hit_truthz"],
                                      "hit_pos":  kn_dict_ak1["hit_pos"],
                                        }),
                        "track":ak.zip({
                                      "x":   kn_dict_ak1["track_x_CAL"],
                                      "y":    kn_dict_ak1["track_y_CAL"],
                                      "ID":    kn_dict_ak1["eventID"],
                                      "pz":    kn_dict_ak1["track_pz_st1"],
                                        }),
                        "st23":ak.zip({
                                      "ntrack23":   kn_dict_ak1["n_st23tracklets"],
                                      "Cal_x":   kn_dict_ak1["st23tracklet_x_CAL"],
                                      "Cal_y":   kn_dict_ak1["st23tracklet_y_CAL"],
                                        })
                       }, depth_limit=1)
    return kn_events
#--------------------------------------------------------------------------------------------------------------------------------------------
def emcal_simhit_selection(arr):
    mask = (arr.detID == 100)
    return mask

def emcal_simhit_selection_energy(arr, e):
    mask = (arr.edep >= e)
    return mask

def h2_selection(arr):
    mask = (arr.detID >= 35) & (arr.detID <= 38)
    return mask

def st2_selection(arr):
    mask = (arr.detID >= 13) & (arr.detID <= 18)
    return mask

def st3_selection(arr):
    mask = (arr.detID >= 19) & (arr.detID <= 30)
    return mask

def h4_selection(arr):
    mask = (arr.detID >= 41) & (arr.detID <= 46)
    return mask
#--------------------------------------------------------------------------------------------------------------------------------------------
ntowersx=72
ntowersy=36
sizex=5.53 # in cm
sizey=5.53 # in cm
ecalx=[-200,200] #size in cm
ecaly=[-100,100]
binsx=ecalx[1]- ecalx[0]
binsy=ecaly[1]- ecaly[0]
sfc = 0.1146337964120158 #sampling fraction of emcal
emin=0.0005

#--------------------------------------------------------------------------------------------------------------------------------------------
def emcal_bytuple(file):
    dq_events = getData(file,"Events")
    x_pos = []#designed to be 2D
    y_pos = []#designed to be 2D
    eve_energy = []#designed to be 2D
    for i in range(len(dq_events[:]["Hits"].edep)):
        output=emcal_byevent(dq_events, i)
        if(len(output[0])!=0):
            x_pos.append(output[0].to_numpy())
            y_pos.append(output[1].to_numpy())
            eve_energy.append(output[2].to_numpy())#here is where pion has 529 wew,wid
    return x_pos, y_pos, eve_energy
#--------------------------------------------------------------------------------------------------------------------------------------------
def emcal_byevent(dq_events,evtNum):
    dq_hits = dq_events[evtNum]["Hits"]
    # select emcal hits
    emcal_mask = emcal_simhit_selection(dq_hits)
    emcal_hits = dq_hits[emcal_mask]
    emcal_energy_mask = emcal_simhit_selection_energy(emcal_hits, emin)
    emcal_hits = emcal_hits[emcal_energy_mask]
    #convert into coordinates and energy_dp
    emcal_towerx = emcal_hits.elmID//ntowersy
    emcal_towery = emcal_hits.elmID%ntowersy
    emcal_x = ecalx[0]+emcal_towerx*sizex
    emcal_y = ecaly[0]+emcal_towery*sizey
    emcal_edep = emcal_hits.edep/sfc
    return emcal_x, emcal_y, emcal_edep

#Clustering return multi cluster xyeng-------------------------------------------------------------------------------------------------------
def label_clus_eng(label, eng_eve):#return sorted labels by decreasing order of energy
    unique_labels = np.unique(label)
    cluster_energy = np.zeros(len(unique_labels), dtype=np.int64)
    for i in range(len(label)):
        cluster_energy[label[i]] += eng_eve[i]
    sorted_labels = unique_labels[np.argsort(cluster_energy)[::-1]]
    return sorted_labels
#--------------------------------------------------------------------------------------------------------------------------------------------
def Clustering_multi(file):
    (x_eve, y_eve, eng_eve)=emcal_bytuple(file)
    labels=[]#2D, for all hits
    centers=[]
    labels_decrease=[]#for each event
    for i in range(len(eng_eve)):
        coordinate=np.column_stack((x_eve[i],y_eve[i]))
        try:
            kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(coordinate)#hardcode
            labels.append(kmeans.labels_)
            labels_decrease.append(label_clus_eng(kmeans.labels_, eng_eve[i]))
            centers.append(kmeans.cluster_centers_)   
        except:
            labels.append([0])
            labels_decrease.append([0])
            centers.append(flatten(coordinate)) 
    return labels, labels_decrease, x_eve, y_eve, eng_eve, centers
#--------------------------------------------------------------------------------------------------------------------------------------------
def cluss_coord(file):#for each cluster, [[x],[y],[eng], center]
    (labels, labels_decrease, x_tup, y_tup, eng_tup, centers_clus)=Clustering_multi(file)
    coord=[]#designed to be 4D(evt-clsuter(decreasing order)-list of xyz- xyz))
    for i in range(len(labels)):
        evt_coord=[]
        for item in labels_decrease[i]:
            
            try:
                indices = np.argwhere(labels[i]==item)
                xvals = flatten(x_tup[i][indices])
                yvals = flatten(y_tup[i][indices])
                energies = flatten(eng_tup[i][indices])
                emax_i=energies.index(max(energies))
                emax_center=[xvals[emax_i], yvals[emax_i]]
            except:
                emax_i=0
                xvals = x_tup[i][emax_i]
                yvals = y_tup[i][emax_i]
                energies = eng_tup[i][emax_i]
                emax_center=[xvals, yvals]

            
            kmeans_center=centers_clus[i][item]
            evt_coord.append([xvals, yvals, energies, emax_center, kmeans_center])#cluster coord in []
        coord.append(evt_coord)
    return coord
#--------------------------------------------------------------------------------------------------------------------------------------------
    
#Clustering return leading cluster xyeng-----------------------------------------------------------------------------------------------------
def find_lead_clus(label, eng_eve):#only return the label for the leading cluster
    count=[0]*(len(np.unique(label)))
    for i in range(len(label)):
        count[label[i]]+=eng_eve[i]
    return np.argmax(count)
#--------------------------------------------------------------------------------------------------------------------------------------------
def Clustering_tuple(file):
    (x_eve, y_eve, eng_eve)=emcal_bytuple(file)
    labels=[]#2D, for all hits
    lead_nums=[]#for each event
    num_clus=[]
    for i in range(len(eng_eve)):
        coordinate=np.column_stack((x_eve[i],y_eve[i]))
        brc = Birch(threshold=20, n_clusters=None)#maximum radius of a cluster is 20, no limit on how much clusters
        brc.fit(coordinate)
        label=brc.predict(coordinate)
        labels.append(label)
        lead_nums.append(find_lead_clus(label, eng_eve[i]))
        num_clus.append(max(label)+1)

    return labels, lead_nums, x_eve, y_eve, eng_eve, num_clus
#--------------------------------------------------------------------------------------------------------------------------------------------
def lead_coord(file):#return the information of leading cluster
    (labels, lead_nums, x_eve, y_eve, eng_eve, num_clus)=Clustering_tuple(file)
    new_x=[]
    new_y=[]
    new_eng=[]
    for i in range(len(lead_nums)):
        x_1d=[]
        y_1d=[]
        eng_1d=[]
        for j, k in enumerate(labels[i]):
            if (k==lead_nums[i]):
                x_1d.append(x_eve[i][j])
                y_1d.append(y_eve[i][j])
                eng_1d.append(eng_eve[i][j])
        new_x.append(x_1d)
        new_y.append(y_1d)
        new_eng.append(eng_1d)
    return new_x, new_y, new_eng#data has been sorted by x order
            
#--------------------------------------------------------------------------------------------------------------------------------------------
def gen_wew(file):
    (x_eve, y_eve, eng_eve)=lead_coord(file)
    wew_x = []
    wew_y = []
    for i in range(len(eng_eve)):
        eng_tot=sum(eng_eve[i])
        x_bar=(np.dot(x_eve[i], eng_eve[i]))/eng_tot
        y_bar=(np.dot(y_eve[i], eng_eve[i]))/eng_tot
        x_sq_eve = []
        y_sq_eve = []
        for j in range(len(eng_eve[i])):
            x_sq_eve.append(eng_eve[i][j]*(x_eve[i][j]-x_bar)**2)
            y_sq_eve.append(eng_eve[i][j]*(y_eve[i][j]-y_bar)**2)
        try: wew_x.append(math.sqrt(sum(x_sq_eve)/eng_tot))
        except ZeroDivisionError: wew_x.append(-1)
        try: wew_y.append(math.sqrt(sum(y_sq_eve)/eng_tot))
        except ZeroDivisionError: wew_y.append(-1)

    return wew_x, wew_y
            
            
#--------------------------------------------------------------------------------------------------------------------------------------------
def gen_wid(file):
    (x_eve, y_eve, eng_eve)=lead_coord(file)
    wid_x = []
    wid_y = []
    for i in range(len(x_eve)):
        try:x_bar=sum(x_eve[i])/len(x_eve[i])
        except ZeroDivisionError: x_bar=0
        try:y_bar=sum(y_eve[i])/len(y_eve[i])
        except ZeroDivisionError: y_bar=0

        x_sq_eve = []
        y_sq_eve = []
        for j in range(len(x_eve[i])):
            x_sq_eve.append((x_eve[i][j]-x_bar)**2)
            y_sq_eve.append((y_eve[i][j]-y_bar)**2)
        try:wid_x.append(math.sqrt(sum(x_sq_eve)/len(x_eve[i])))
        except ZeroDivisionError: wid_x.append(-1)
        try:wid_y.append(math.sqrt(sum(y_sq_eve)/len(y_eve[i])))
        except ZeroDivisionError: wid_y.append(-1)
    return wid_x, wid_y
# --------------------------------------------------------------------------------------------------------------------------------------------
def gen_skew(file):
    skew_x=[]
    skew_y=[]
    (x,y,eng)=lead_coord(file)
    for i in range(len(eng)):
        skew_x.append(sp.skew(x[i]))
        skew_y.append(sp.skew(y[i]))
    return np.nan_to_num(np.hstack(skew_x), nan=-9999), np.nan_to_num(np.hstack(skew_y), nan=-9999)

#--------------------------------------------------------------------------------------------------------------------------------------------
def gen_kurt(file):
    kurt_x=[]
    kurt_y=[]
    (x,y,eng)=lead_coord(file)
    for i in range(len(eng)):
        kurt_x.append(sp.skew(x[i]))
        kurt_y.append(sp.skew(y[i]))
    return np.nan_to_num(np.hstack(kurt_x), nan=-9999), np.nan_to_num(np.hstack(kurt_y), nan=-9999)

#--------------------------------------------------------------------------------------------------------------------------------------------
def ntrack_st23(file):
    dq_events = getData(file,"Events")
    return dq_events["st23"].ntrack23.tolist()
#--------------------------------------------------------------------------------------------------------------------------------------------
#finding the energy weighted and geometric center
def gen_center(file):
    output=lead_coord(file)
    x_eve=output[0]
    y_eve=output[1]
    eng_eve=output[2]
    wew_x = []
    wew_y = []
    #wid_x = []
    #wid_y = []
    for i in range(len(eng_eve)):
        eng_tot=sum(eng_eve[i])
        wew_x.append(np.dot(x_eve[i], eng_eve[i])/eng_tot)
        wew_y.append(np.dot(y_eve[i], eng_eve[i])/eng_tot)
        #try:wid_x.append(sum(x_eve[i])/len(x_eve[i]))
        #except: wid_x.append([])
        #try:wid_y.append(sum(y_eve[i])/len(y_eve[i]))
        #except: wid_y.append([])
    return wew_x, wew_y   
#--------------------------------------------------------------------------------------------------------------------------------------------
def track_bytuple(file):
    dq_events = getData(file,"Events")
    track_x = dq_events["track"].x.tolist()
    track_y = dq_events["track"].y.tolist()
    return track_x, track_y
#--------------------------------------------------------------------------------------------------------------------------------------------
def track_cal(file):
    output1 = gen_center(file)
    output2 = valid_track(file)# <Array [] type='0 * float32'>, some entries are empty
    wew_x = output1[0]
    wew_y = output1[1]
    track_x = output2[0]
    track_y = output2[1]
    diff_x = []
    diff_y = []
    for i in range(len(track_x)):
        diff_x.append(track_x[i]-wew_x[i])
        diff_y.append(track_y[i]-wew_y[i])
    return np.hstack(diff_x), np.hstack(diff_y)
#--------------------------------------------------------------------------------------------------------------------------------------------
def count(list1, x1, x2):
    return len([x for x in list1 if x >= x1 and x <= x2])
#--------------------------------------------------------------------------------------------------------------------------------------------
def flatten(l):#unfold tuple, decrease by 1D
    return [item for sublist in l for item in sublist]
#--------------------------------------------------------------------------------------------------------------------------------------------
def gen_Ep(file):
    p=np.hstack(valid_track(file)[2])
    raw_e=emcal_bytuple(file)[2]
    e=[]
    for item in raw_e:
        e.append(sum(item))
    Ep=[]
    for i in range(len(p)):
        if p[i]<0:
            Ep.append(-1)
        else:
            Ep.append(e[i]/p[i])
    return Ep
#--------------------------------------------------------------------------------------------------------------------------------------------
def valid_track(file):
    dq_events = getData(file,"Events")
    track_x = dq_events["track"].x.tolist()
    track_y = dq_events["track"].y.tolist()
    p=dq_events["track"].pz.tolist()
    x1,y1=gen_center(file)

    #fill empty with -1, check multi points
    indecies=[]
    for i in range(len(p)):
        if p[i]==[]:
            p[i].append(-1)
            track_x[i].append(-1)
            track_y[i].append(-1)
        elif len(p[i])!=1:
            indecies.append(i)
    #pop out wrong tracks
    if len(indecies)!=0:
        for i in range(len(indecies)):
            pt= [x1[indecies[i]], y1[indecies[i]]]
            coords= list(zip(track_x[indecies[i]], track_y[indecies[i]]))
            distance,index = spatial.KDTree(coords).query(pt)
            track_x[indecies[i]]=track_x[indecies[i]][index]
            track_y[indecies[i]]=track_y[indecies[i]][index]
            p[indecies[i]]=p[indecies[i]][index]
    return track_x, track_y, p
#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------

