'''
Author:          Victor Mireles - Freie Universitaet Berlin
Email:           syats.vm@gmail.com

Latest version:  https://github.com/syats/ModuleReusability

Distributed under GPL V3.0
'''


"""
	Takes as input a list of decomposableMatrix objects. Each of them represents a decomposition into K modules
"""

def histogram_of_module_sizes(decompositions_for_single_k,n=None,r=None):
	k = decompositions_for_single_k[0].theMatrix.shape[1]
	for i,dec in enumerate(decompositions_for_single_k):
		if dec.theMatrix.shape[1] != k:
			print("decomposition "+str(i)+" has the wrong number of modules")
			raise ValueError



import numpy as np
import scipy as sci
import gradDescent.heuristic5DMandList as h1
import aux.auxFunctionsGEO as aGEO
import permutationsOfArtificialModules as pam
import matplotlib.pyplot as plt
import plottingWithRandEquivalents as  pt
import aux.decompositionWithSurrogates as dws
from   gradDescent.measuresForDecomposableMatrix import sizeMeasures, perModuleReusability
import aux.auxPreprocessing as app
import os
import matplotlib.pyplot as plt



# -------- THE DATA --------------------------------
filePath   = 'BrawandData/';  #From Brawand et al. Nature 2011.   doi:10.1038/nature10532
#fileNames  = ['Human_Ensembl57_TopHat_UniqueReads.txt',
#				'Bonobo_Ensembl57_TopHat_UniqueReads.txt', 'Chicken_Ensembl57_TopHat_UniqueReads.txt']  #EG

fileNames  = ['Chimpanzee_Ensembl57_TopHat_UniqueReads.txt', 'Gorilla_Ensembl57_TopHat_UniqueReads.txt', 'Macaque_Ensembl57_TopHat_UniqueReads.txt', 'Mouse_Ensembl57_TopHat_UniqueReads.txt']  #Hell9000
#fileNames  = [''Opossum_Ensembl57_TopHat_UniqueReads.txt', 'Orangutan_Ensembl57_TopHat_UniqueReads.txt', 'Platypus_Ensembl57_TopHat_UniqueReads.txt' ]  #lnsyc

treatAs   = ['CSV' for x in fileNames]

# For test data -->
filePath = "testData/GPL13987/"
fileNames =  ["GSE33045fluid_.dat", "GSE33045plasma_.dat"]
treatAs = ['GEO' for x in fileNames]


GEOthr    = 35
CSVthr    = 1
CSVsep	  = '\t'
CSVreadStart = 6;
#GEO treatment expects number of PCR cycles, GEOthr or more is considered AS NOT FOUND. This file format does not expect column or row titles
#CSV treatment expects Integer copy numbers / read counts, CSVthr or more is considered AS FOUND. This file format expects both column and row titles. The row titles are the gene names, for example, and are expected to be the first column of each row. The actual data is considered to be in columns CSVreadStart on wards (0-based)

# ------ THE RANDOM EQUIVALENTS
numRand = 2; 		 # Number of DP-Rand matrices to compare to
numRSS3 = 2; 		 # Numbre of RSS-Rand to compare to

storageDir = 'storage/';
#The computations are saved here, not the decompositions themselves but only the data necesary for recomputing the plots. The filenames are a hash of the input matrix, and the files contain pickled objects of class decompositionWithSurrogates. If something changes and you want to rerun everything for a given input file, you must delete the corresponding file in the storageDir

toSaveDirs   = [None for x in fileNames]
#One can also save the decompositions of the real data... beware it can take up lots of space, None means the decompositions aren't saved

# ---------------------------------------


# -----------  WHAT TO COMPUTE WITH THE DATA  ----------------------------
#First entry: size (0) or reusability (1)
#Second entry   mean(0) maximum (2) or entropy (2)
toPlot = [(0,0),(0,1),(1,2),(1,0)]
titles = ['Mean size \nAUC ratio ','Max size \nAUC ratio','entropy of reusability \nAUC ratio ','Mean reusability \nAUC ratio']
# ---------------------------------------


# -------- THE HEURISITC PARAMETERS, See Mireles & Conrad 2015 for full meaning -----
# D+Q is the total number of k+1 decomps to make for each k decomp so far computed, D are "random" (exploration)  decomps and Q are greedy (explotation)
# L is the number of k decomp to save for the next iteration (only the best, the rest are discarded)
# W + L is how many k's are the L*(D+Q) decompositions propagated greedily before discarding any of them
# R is the last k for which decomps are computed. If negative, all possible k's are computed

heuristicParms={'Q':1 , 'l':2 , 'w': 0, 'r':-1 , 'D':1}  # Fast for testing
heuristicParms={'Q':2 , 'l':4 , 'w': 2, 'r':-1 , 'D':8}  # (5X) Slower for real analysis
nc = 4			 # Number of cores to run the heuristic algorithm on
# ---------------------------------------



#Make the decompositions both of the real data and of its random equivalents.
#for fnNUM,fn in enumerate(fileNames):
fnNUM, fn= 0, fileNames[0]

print("\nFN:" + fn)

saveDir = toSaveDirs[fnNUM]
savePrefix = None
if saveDir is not None:
	savePrefix = fn.split("_")[0]

fileName = os.path.join(filePath,fn)
if treatAs[fnNUM]=='GEO':
	C,QC = aGEO.load_TaqMan_csv(fileName,thr=GEOthr)
	gNames = ""
if treatAs[fnNUM]=='CSV':
	C,QC,gNames = aGEO.load_Labeled_csv(fileName,thr=CSVthr,sep=CSVsep,readStart=CSVreadStart)

m = max(C.shape)
n = min(C.shape)
emptyGenes,detr,translationOfRows,Cs = app.cleanInputMatrix(C,alsoReturnTranslationToOriginal=True)
print("\t"+str(C.shape)+" -> "+str(Cs.shape))
nodeWeights = np.zeros(max(Cs.shape))
for x in range(m):
	iind = int(translationOfRows[x])
	if iind >= 0:
		nodeWeights[iind] += 1
r = max(Cs.shape)

all_dec = h1.heuristic1(C=Cs.T,
						n=n,m=m,
						outputDataPathO=saveDir,
						outputDataPrefixO=savePrefix,
						numCores = nc,
						numGreedy = heuristicParms['w'],
						geneWeights = nodeWeights,
						returnList=True,
						display=saveDir!=None,
						return_all_decomps=True,
						**heuristicParms)

#for k in range(n,r):
for k in [n+nr*10 for nr in range(100) if nr*10 < r]+[r]:
	# for each found value of resusability, data_for_heatmap will be a dictionary of type:
	#   module_size:  fraction_of_modules
	# e.g. to see what is the  fraction of modules of size s in a decompositoin with
	# reusability r,  consult   data_for_heatmap[r][s]
	# end num_per_reuse will count how many decompositions have this value of reusability
	counts_for_heatmap = dict()
	num_per_reuse = dict()
	decos_this_k = [x for x in all_dec if x.theMatrix.shape[1]==k]
	for deco in decos_this_k:
		sizes = deco.theSizes
		reuse = perModuleReusability(deco, C)[0].mean()
		num_per_reuse[reuse] = num_per_reuse.get(reuse, 0) + 1
		if not reuse in counts_for_heatmap.keys():
			counts_for_heatmap[reuse] = {s:[] for s in range(1, m + 1)}
		for size in set(sizes):
			frac_this_size = len([s for s in sizes if s == size]) / float(k)
			if frac_this_size == 0:
				continue
			counts_for_heatmap[reuse][size] = counts_for_heatmap[reuse].get(size, []) + [frac_this_size]

	# At the end we divide by num_per_reuse so that  data_for_heatmap[r][s] is the average
	# fraction of blocks of size s in decompositions of reusability r

	if len(counts_for_heatmap)==0:
		continue
	means_for_heatmap=dict()
	stds_for_heatmap = dict()
	totmax_size = -np.inf
	totmin_size = np.inf
	for reuse in counts_for_heatmap.keys():
		if reuse not in means_for_heatmap.keys():
			means_for_heatmap[reuse] = dict()
			stds_for_heatmap[reuse] = dict()
		for size in counts_for_heatmap[reuse].keys():
			if len(counts_for_heatmap[reuse][size]) > 0:
				if size > totmax_size:
					totmax_size = size
				if size < totmin_size:
					totmin_size = size
				means_for_heatmap[reuse][size] = np.mean(counts_for_heatmap[reuse][size])
				stds_for_heatmap[reuse][size] = np.std(counts_for_heatmap[reuse][size])

	reuses = list(counts_for_heatmap.keys())
	reuses.sort()
	numreuses = len(reuses)
	numsizes = len(range(totmin_size, totmax_size+1))
	heatmap_data = np.zeros((numreuses, numsizes))
	for rn,reuse in enumerate(reuses):
		for size in means_for_heatmap[reuse].keys():
			heatmap_data[rn, size-totmin_size] = means_for_heatmap[reuse][size]

	plt.figure(k)
	plt.imshow(heatmap_data, cmap='hot', interpolation='none', aspect='auto')
	plt.yticks(list(range(numreuses)), ["%.2f" % a for a in reuses])
	plt.ylabel("Mean decomposition reusability ")
	plt.xlabel("PBB size ")
	plt.colorbar()

plt.show()

