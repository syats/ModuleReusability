'''
Author:          Victor Mireles - Freie Universitaet Berlin
Email:           syats.vm@gmail.com

Latest version:  https://github.com/syats/ModuleReusability

Distributed under GPL V3.0
'''

"""
	Takes as input a list of decomposableMatrix objects. Each of them represents a decomposition into K modules
"""

import numpy as np
import pickle
import gradDescent.heuristic5DMandList as h1
import aux.auxFunctionsGEO as aGEO
import aux.auxPreprocessing as app
import os
import math
import matplotlib.pyplot as plt
import aux.aux_stats as aast
from gradDescent.measuresForDecomposableMatrix import sizeMeasures, perModuleReusability
import plot_functions as pf

# For test data -->
filePath = "/home/syats/Modularity/data_sets/GPL13987"
fileNames = ['GSE48908_non-normalized_data_KMS11.csv',
             'GSE37766_non_normalized_cardA.csv',
             'GSE45387_SUDHL4_non-normalized.csv',
             'GSE48909_non-normalized_data_RPMI8226.csv',
             "GSE47652_non_normalized.csv"]
treatAs = ['GEO' for x in fileNames]

fnn = 4  # <--- From the list of fileNames, which to take
plotHeatmaps = False
plotEntropies = False
plotDeltaMus = False
plotTriptych = True
binwidth_for_sizes = 5
numbins_for_reuses = 15
numbins_for_entropies = 15
compute = False
append = True
k_step_size = 4
plot_each_k = False
division_pos = 120

maximum_r = 30
num_reruns = 1

# ---------------------------------------------------------------
GEOthr = 35
CSVthr = 1
CSVsep = '\t'
CSVreadStart = 6;
# GEO treatment expects number of PCR cycles, GEOthr or more is considered AS NOT FOUND. This file format does not expect column or row titles
# CSV treatment expects Integer copy numbers / read counts, CSVthr or more is considered AS FOUND. This file format expects both column and row titles. The row titles are the gene names, for example, and are expected to be the first column of each row. The actual data is considered to be in columns CSVreadStart on wards (0-based)

storageDir = 'storage/';
# The computations are saved here, not the decompositions themselves but only the data necesary for recomputing the plots. The filenames are a hash of the input matrix, and the files contain pickled objects of class decompositionWithSurrogates. If something changes and you want to rerun everything for a given input file, you must delete the corresponding file in the storageDir

toSaveDirs = [None for x in fileNames]
# One can also save the decompositions of the real data... beware it can take up lots of space, None means the decompositions aren't saved

# ---------------------------------------

# -------- THE HEURISITC PARAMETERS, See Mireles & Conrad 2015 for full meaning -----
# D+Q is the total number of k+1 decomps to make for each k decomp so far computed, D are "random" (exploration)  decomps and Q are greedy (explotation)
# L is the number of k decomp to save for the next iteration (only the best, the rest are discarded)
# W + L is how many k's are the L*(D+Q) decompositions propagated greedily before discarding any of them
# R is the last k for which decomps are computed. If negative, all possible k's are computed

heuristicParms = {'Q': 1, 'l': 2, 'w': 1, 'r': -1, 'D': 3}
computing_heuristicParms = {'Q': 4, 'l': 10, 'w': 2, 'r': maximum_r, 'D': 20}
nc = 4  # Number of cores to run the heuristic algorithm on
# ---------------------------------------


# Make the decompositions both of the real data and of its random equivalents.
# for fnNUM,fn in enumerate(fileNames):
fnNUM, fn = 0, fileNames[fnn]

print("\nFN:" + fn)

saveDir = toSaveDirs[fnNUM]
savePrefix = None
if saveDir is not None:
    savePrefix = fn.split("_")[0]

fileName = os.path.join(filePath, fn)
if treatAs[fnNUM] == 'GEO':
    C, QC = aGEO.load_TaqMan_csv(fileName, thr=GEOthr)
    gNames = ""
if treatAs[fnNUM] == 'CSV':
    C, QC, gNames = aGEO.load_Labeled_csv(fileName, thr=CSVthr, sep=CSVsep, readStart=CSVreadStart)

m = max(C.shape)
n = min(C.shape)
emptyGenes, detr, translationOfRows, Cs = app.cleanInputMatrix(C, alsoReturnTranslationToOriginal=True)
print("\t" + str(C.shape) + " -> " + str(Cs.shape))
nodeWeights = np.zeros(max(Cs.shape))
for x in range(m):
    iind = int(translationOfRows[x])
    if iind >= 0:
        nodeWeights[iind] += 1
r = max(Cs.shape)

if 'pre_computed' not in locals():
    pre_computed = set()
all_dec = set()

if (append or (not compute)) and (len(pre_computed) == 0):
    print("loading.... ")
    try:
        pre_computed = pickle.load(open(os.path.join(storageDir, fn + "decompos.pkl"), "rb"))
        pre_computed = set(pre_computed)
    except:
        print("File not found")
        compute = True
    print("loaded:  " + str(len(pre_computed)) + " decos")
    pre_computed = list(pre_computed)
    for dec in pre_computed:
        dec.orderColumns()
    pre_computed = set(pre_computed)
    print("loaded:  " + str(len(pre_computed)) + " decos after re-sorting")

if compute:
    heuristicParms = computing_heuristicParms
if compute or append:
    print("computing....")
    for rerun in range(num_reruns):
        print("\t"+str(rerun+1)+ "/" + str(num_reruns))
        all_dec = all_dec | h1.heuristic1(C=Cs.T,
                            n=n, m=m,
                            outputDataPathO=saveDir,
                            outputDataPrefixO=savePrefix,
                            numCores=nc,
                            numGreedy=heuristicParms['w'],
                            geneWeights=nodeWeights,
                            returnList=True,
                            display=saveDir != None,
                            return_all_decomps=True,
                            **heuristicParms)

    print("computed " + str(len(all_dec)) + ". Now saving....")
    all_dec = list(all_dec | pre_computed)
    for dec in all_dec:
        dec.orderColumns()
    all_dec = set(all_dec)
    pickle.dump(all_dec, open(os.path.join(storageDir, fn + "decompos.pkl"), "wb"))
    print("Finished computing & saving.")

all_dec = all_dec | pre_computed

print("Number of decos:  " + str(len(all_dec)))



# ---------- PREPARE DATA FOR PLOTTING ------------------
allks = [1 + n + nr * k_step_size
         for nr in range(100)
         if 1 + n + nr * k_step_size < r]
allks = [n + 1 + i for i in range(20)]
allks = [20,26,31,36,46,51]
allks = range(23,37)

all_reuses_and_size_entropies = dict()
all_reuses_and_delta_mus = dict()
all_heatmap_data_size_reuse = dict()
all_heatmap_data_entr_reuse = dict()
all_reuse_bins = dict()
all_entr_bins = dict()
all_reuse_hists = dict()
min_sizes = dict()
all_large_reuse_hists = dict()
all_small_reuse_hists = dict()

for k in allks:
    # for each found value of resusability, data_for_heatmap will be a dictionary of type:
    #   module_size:  fraction_of_modules
    # e.g. to see what is the  fraction of modules of size s in a decompositoin with
    # reusability r,  consult   data_for_heatmap[r][s]
    # end num_per_reuse will count how many decompositions have this value of reusability

    counts_for_heatmap_size_reuse = dict()
    counts_for_heatmap_entr_reuse = dict()
    num_per_reuse = dict()
    decos_this_k = [x for x in all_dec if x.theMatrix.shape[1] == k]
    reuses_and_size_entropies = []
    reuses_and_delta_mus = []
    reuses_all_decos = np.zeros(len(decos_this_k))
    reuses_large_modules = dict()
    reuses_small_modules = dict()
    reuses_large_binned = dict()
    reuses_small_binned = dict()

    for decn, deco in enumerate(decos_this_k):
        sizes = deco.theSizes
        per_module_reuse = perModuleReusability(deco, Cs)[0]
        reuse = per_module_reuse.mean()
        reuses_all_decos[decn] = reuse
        sizes_hist = np.histogram(sizes,
                                  [binwidth_for_sizes * i
                                   for i in range(m)
                                   if binwidth_for_sizes * i <= m])[0]
        sizes_hist = [x / float(k) for x in sizes_hist]
        # Shannon's Entropy in Bits
        H = -sum([x * (math.log(x) / math.log(2) if x > 0 else 0) for x in sizes_hist])
        reuses_and_size_entropies.append((reuse, H))

        num_per_reuse[reuse] = num_per_reuse.get(reuse, 0) + 1
        if not reuse in counts_for_heatmap_size_reuse.keys():
            counts_for_heatmap_size_reuse[reuse] = {s: [] for s in range(1, m + 1)}
            counts_for_heatmap_entr_reuse[reuse] = {}
            reuses_large_modules[reuse] = np.zeros(n + 1)
            reuses_small_modules[reuse] = np.zeros(n + 1)
        for size in set(sizes):
            frac_this_size = len([s for s in sizes if s == size]) / float(k)
            if frac_this_size == 0:
                continue
            counts_for_heatmap_size_reuse[reuse][size] = counts_for_heatmap_size_reuse[reuse].get(size, []) + [
                frac_this_size]
        counts_for_heatmap_entr_reuse[reuse][H] = counts_for_heatmap_entr_reuse[reuse].get(H, 0) + 1

        u = (np.array(sizes) - min(sizes)) / float(m)
        mu1, mu2 = aast.find_two_maxima(u, ax=0)
        mu1 = mu1 * m + min(sizes)
        mu2 = mu2 * m + min(sizes)
        reuses_and_delta_mus.append((reuse, np.abs(mu1 - mu2)))

        reuse_large_modules = [per_module_reuse[i] for i in range(len(sizes))
                               if sizes[i] >= division_pos]
        reuse_small_modules = [per_module_reuse[i] for i in range(len(sizes))
                               if sizes[i] < division_pos]

        for i in range(n+1):
            reuses_large_modules[reuse][i] += len([x for x in reuse_large_modules if x == i])
            reuses_small_modules[reuse][i] += len([x for x in reuse_small_modules if x == i])


    # At the end we divide by num_per_reuse so that  data_for_heatmap[r][s] is the average
    # fraction of blocks of size s in decompositions of reusability r

    if len(counts_for_heatmap_size_reuse) == 0:
        continue
    reuses_this_k = list(counts_for_heatmap_size_reuse.keys())
    re = np.linspace(min(reuses_this_k), max(reuses_this_k), numbins_for_reuses)
    reuse_bins = np.linspace(min(reuses_this_k), max(reuses_this_k), numbins_for_reuses)
    reuse_hist = np.histogram(reuses_all_decos,reuse_bins)[0]
    entropies_this_k = [x[1] for x in reuses_and_size_entropies]
    entr_bins = np.linspace(min(entropies_this_k), max(entropies_this_k), numbins_for_entropies)

    means_for_heatmap = dict()
    stds_for_heatmap = dict()
    sums_for_heatmap_size_reuse = dict()
    heatmap_data_entr_reuse = np.zeros((numbins_for_reuses, numbins_for_entropies))

    totmax_size = -np.inf
    totmin_size = np.inf

    # Discretize the reuses into  reusebins ---
    # and the entropies into entrbins
    for reuse in reuses_this_k:
        rn = max([0, reuse_bins.searchsorted([reuse])[0] - 1])
        if rn not in means_for_heatmap.keys():
            means_for_heatmap[rn] = dict()
            stds_for_heatmap[rn] = dict()
            sums_for_heatmap_size_reuse[rn] = dict()
            reuses_large_binned[rn] = np.zeros(n + 1)
            reuses_small_binned[rn]= np.zeros(n + 1)
        for size in counts_for_heatmap_size_reuse[reuse].keys():
            if len(counts_for_heatmap_size_reuse[reuse][size]) > 0:
                if size > totmax_size:
                    totmax_size = size
                if size < totmin_size:
                    totmin_size = size
                means_for_heatmap[rn][size] = np.mean(counts_for_heatmap_size_reuse[reuse][size])
                stds_for_heatmap[rn][size] = np.std(counts_for_heatmap_size_reuse[reuse][size])
                sums_for_heatmap_size_reuse[rn][size] = k * np.sum(counts_for_heatmap_size_reuse[reuse][size])
        for h in counts_for_heatmap_entr_reuse[reuse].keys():
            hn = max([0, entr_bins.searchsorted([h])[0] - 1])
            heatmap_data_entr_reuse[rn][hn] += counts_for_heatmap_entr_reuse[reuse][h]
        reuses_large_binned[rn] += reuses_large_modules[reuse]
        reuses_small_binned[rn] += reuses_small_modules[reuse]

    reuses_this_k.sort()
    numreuses = len(reuses_this_k)
    numsizes = len(range(totmin_size, totmax_size + 1))
    heatmap_data_size_reuse = np.zeros((numbins_for_reuses, numsizes))

    # After entering all the counts, we normalize row-wise the matrices for the heatmaps
    for rn in range(numbins_for_reuses):
        if not rn in sums_for_heatmap_size_reuse.keys():
            continue
        tot_this_row = sum(sums_for_heatmap_size_reuse[rn].values())
        for size in sums_for_heatmap_size_reuse[rn].keys():
            heatmap_data_size_reuse[rn, size - totmin_size] = sums_for_heatmap_size_reuse[rn][size] / tot_this_row
        sum_this_entr = heatmap_data_entr_reuse[rn,:].sum()
        heatmap_data_entr_reuse[rn,:] = heatmap_data_entr_reuse[rn,:] / sum_this_entr

    min_sizes[k] = totmin_size
    all_heatmap_data_size_reuse[k] = heatmap_data_size_reuse
    all_heatmap_data_entr_reuse[k] = heatmap_data_entr_reuse
    all_reuses_and_delta_mus[k] = reuses_and_delta_mus
    all_reuses_and_size_entropies[k] = reuses_and_size_entropies
    all_reuse_bins[k] = reuse_bins
    all_reuse_hists[k] = reuse_hist
    all_entr_bins[k] = entr_bins
    all_small_reuse_hists[k] = reuses_small_binned
    all_large_reuse_hists[k] = reuses_large_binned

# ----- Do the plotting ----------------------
if plotTriptych:
    pf.heatmap_triptych(all_heatmap_data_size_reuse, all_reuse_bins, min_sizes, all_reuse_hists,
                        all_large_reuse_hists, all_small_reuse_hists)

if plotHeatmaps:
    pf.heatmap_reuse_vs_size(all_heatmap_data_size_reuse, all_reuse_bins, min_sizes,
                             all_heatmap_data_entr_reuse, all_entr_bins, all_reuse_hists,
                             size_compress_factor=3)
if plotEntropies:
    pf.scatter_reuse_vs_entropiesSizes(all_reuses_and_size_entropies, plot_each=plot_each_k)
if plotDeltaMus:
    pf.scatter_reuse_vs_muDistance(all_reuses_and_delta_mus, plot_each=plot_each_k)

plt.show()
