'''
Author:          Victor Mireles - Freie Universitaet Berlin
Email:           syats.vm@gmail.com

Latest version:  https://github.com/syats/ModuleReusability

Distributed under GPL V3.0
'''

import numpy as np
import scipy as sci
import aux.auxFunctionsGEO as aGEO
import permutationsOfArtificialModules as pam
import matplotlib.pyplot as plt
import plottingWithRandEquivalents as  pt
import aux.decompositionWithSurrogates as dws
from   gradDescent.measuresForDecomposableMatrix import sizeMeasures
import aux.auxPreprocessing as app


# -------- THE DATA --------------------------------
filePath   = 'testData/';
fileNames  = ['smallSample.csv','GPL13987/GSE33045fluid_.dat','GPL13987/GSE33045plasma_.dat']
treatAs   = ['CSV','GEO','GEO']
GEOthr    = 35
CSVthr    = 1
CSVsep	  = '\t'
#GEO treatment expects number of PCR cycles, GEOthr or more is considered AS NOT FOUND. This file format does not expect column or row titles
#CSV treatment expects Integer copy numbers / read counts, CSVthr or more is considered AS FOUND. This file format expects both column and row titles. The row titles are the gene names, for example

# ------ THE RANDOM EQUIVALENTS
numRand = 2; 		 # Number of DP-Rand matrices to compare to
numRSS3 = 2; 		 # Numbre of RSS-Rand to compare to

storageDir = 'storage/';
#The computations are saved here, not the decompositions themselves but only the data necesary for recomputing the plots. The filenames are a hash of the input matrix, and the files contain pickled objects of class decompositionWithSurrogates. If something changes and you want to rerun everything for a given input file, you must delete the corresponding file in the storageDir
toSaveDirs = ['decompositions/',None,None]
#One can also save the decompositions of the real data... beware it can take up lots of space

# ---------------------------------------


# -----------  WHAT TO COMPUTE WITH THE DATA  ----------------------------
#First entry: size (0) or reusability (1)
#Second entry   mean(1) maximum (2) or entropy (2)
toPlot = [(0,0),(0,1),(1,2)]
titles = ['Mean size \nAUC ratio ','Max size \nAUC ratio','entropy of reusability \nAUC ratio ']
# ---------------------------------------


# -------- THE HEURISITC PARAMETERS, See Mireles & Conrad 2015 for full meaning -----
# D+Q is the total number of k+1 decomps to make for each k decomp so far computed, D are "random" (exploration)  decomps and D are greedy (explotation)
# L is the number of k decomp to save for the next iteration (only the best, the rest are discarded)
# W + L is how many k's are the L*(D+Q) decompositions propagated greedily before discarding any of them

heuristicParms={'Q':2 , 'L':4 , 'W': 2, 'R':-1 , 'D':6}
heuristicParms={'Q':1 , 'L':1 , 'W': 1, 'R':-1 , 'D':2}
nc = 4; 			 # Number of cores to run the heuristic algorithm on
# ---------------------------------------


#Do the computations ------------------------------------------------

dataS  = []
randSs = []
rss3Ss = []
allData = dict()
origCs  = dict();
origQCs = dict();
origGNs = dict()
dataTypes = dict()
dataNames = dict()
KvsSizeHists = dict();
allStores = dict()

#Make the decompositions both of the real data and of its random equivalents.
for fnNUM in range(len(fileNames)):
	fn = fileNames[fnNUM];

	saveDir = toSaveDirs[fnNUM];
	savePrefix = None;
	if saveDir != None:
		savePrefix = fn;

	print("\nFN:" + fn)
	thisNumRand = numRand;
	thisNumRSS3 = numRSS3;
	fileName = filePath+fn;
	if treatAs[fnNUM]=='GEO':
		C,QC = aGEO.load_TaqMan_csv(fileName,thr=GEOthr);
		gNames = "";
	if treatAs[fnNUM]=='CSV':
		C,QC,gNames = aGEO.load_Labeled_csv(fileName,thr=CSVthr,sep=CSVsep);
	hashC = hash(str(C.tolist()));
	totHash = str(hashC)+"_"+str(C.shape[0])+"."+str(C.shape[1])+"_"+str(C.sum());

	dataThisFn,typesThisFn,toStore = dws.loadOrCompute(fn,C,storageDir,
						totHash,
						heuristicParms,
						['SRpairs'] ,
						thisNumRSS3O=thisNumRSS3,
						thisNumRandO=thisNumRand,
						nc=nc,
						saveDir=saveDir,savePrefix=savePrefix);

	allStores[fn] = toStore;
	allData[fn]   = dataThisFn;
	dataTypes[fn] = typesThisFn;
	dataNames[fn] = str.split(fn,'_')[0]
	origCs[fn]    = C;
	origQCs[fn]   = QC;
	origGNs[fn]   = gNames;

#Make the actual plots

plt.close()
fig1 = plt.figure()
fig2 = plt.figure()
spn = 1;
spn2 = 1;
for sizeOrReuse,MeanMaxOrEnt in toPlot:
	dataToPlot = dws.flattenSizeReuseList(allData,origCs,sizeOrReuse,MeanMaxOrEnt)

	#Violin plots
	ax = fig1.add_subplot(len(toPlot),1,spn);
	dataFV = pt.plotWithReasAsBaes(dataToPlot,dataTypes,origCs,namesOfData=dataNames)
	pt.makeViolinPlots(dataFV,theAx=ax,ytitle=titles[spn-1])
	if spn < len(toPlot):
		ax.set_xticks([])

	#Band plots
	for fn in dataToPlot.keys():
		ax = fig2.add_subplot(len(toPlot),len(dataToPlot),spn2);
		singletonDict = {fn:dataToPlot[fn]}
		pt.makeBands(singletonDict,dataTypes,origCs,namesOfData=dataNames,avgY=False,logY=False,plotDataSums=True,QCs=origQCs,numStds=1,theAx = ax,yLabel=titles[spn-1].split("\n")[0],finalK=len(dataToPlot[fn][0]),legendPos=None)
		#Top row
		if spn2 <= len(dataToPlot):
			ax.set_title(dataNames[fn])
		#Not left col
		if (spn2-1) % len(dataToPlot) > 0:
			ax.set_yticks([])
			ax.set_ylabel("")
		spn2+=1

	spn += 1

fig1.subplots_adjust(wspace=0.079,hspace=0.0,left=0.14,right=0.99,top=0.92,bottom=0.27)
fig2.subplots_adjust(hspace=0.0,wspace=0,left=0.14,right=0.99,top=0.92,bottom=0.27)
plt.show()
