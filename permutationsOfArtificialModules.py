'''
Author:          Victor Mireles - Freie Universitaet Berlin
Email:           syats.vm@gmail.com

Latest version:  https://github.com/syats/ModuleReusability

Distributed under GPL V3.0
'''

import numpy as np
import itertools as it
import random as ran

import gradDescent.extendFromRtoM as eRM
import aux.auxSetFunctions as sf
import gradDescent.heuristic5DMandList as h1
import aux.generateSimpleMatrix as agm
import aux.auxPreprocessing as app
import gradDescent.measuresForDecomposableMatrix as MD
import gradDescent.decomposableMatrix as DM

# m: rows   n: columns    nnz:non zero entries,
# mods=[[c1,r1],[c2,r2], .... , [cl,rl]]  sizes of l modules we want embeded.
#c1+c2+...+cl < n     r1+r2+.....+rl < m
# c1*r1 + c2*r2 + .... + cl*rl  < nnz

def createMatrix(n,m,nnz,mods):
	de = nnz - sum([(mod[0]*mod[1]) for mod in mods]);
	ms = m   - sum([mod[1] for mod in mods]);
	repeat = 1;
	while(repeat > 0):
		repeat = 0;
		M = agm.generateRrepetitions(n,ms,de);
		# Ctemp is missing rows where the modules should go. For every module of size [ci,ri] we will add ri rows, each of which will have it's first ci entries as 1's and the rest as 0's
		Mtemp = np.max(M.shape);
		transls,Cs = app.cleanInputMatrix(M);
		if min(Cs.shape) < n:
			repeat = 1;

	for mod in mods:
		row = np.zeros([1,n]);
		row[0,0:mod[0]] = 1;
		subM = np.repeat(row,mod[1],axis=0);

		M = np.vstack([subM,M]);

	return M

def createRSS2Equivalent(C):
	translationOfRows,Cs = app.cleanInputMatrix(C);
	Ct = createRSS3Equivalent(Cs);
	Ct = DM.decomposableMatrix(Ct)
	trans,Cr = eRM.expandMatrix(Ct,translationOfRows=translationOfRows);
	CR = np.array(Cr.theMatrix.todense());
	sumCR = CR.sum(axis=1);
	ze = np.nonzero(sumCR==0)[0];
	for z in ze:
		i = ran.sample(range(min(C.shape)),1)[0];
		CR[z,i] = 1;
	return CR;

def createRSS3Equivalent(C):
	return agm.generateRSS(C)

def createUniformRowSumEquivalent(C):
	reload(agm)
	return agm.generateWithUniformRows(C);

def createRANDEquivalent(origC):
	de = origC.sum();
	return agm.generateRrepetitions(np.min(origC.shape),np.max(origC.shape),de);


def decomposeMatrixAndGetSumOfVariances(Cz,returnListOfVars=False,nCores=1,heuristicParameters={'Q':2 , 'L':4 , 'W': 2, 'R':-1 , 'D':6},returnlistOfsizeMeasures=True,saveDir=None,savePrefix=None):

	Q = heuristicParameters['Q'];  #2
	L = heuristicParameters['L'];  #4
	W = heuristicParameters['W'];  #2
	R = heuristicParameters['R'];  #-1
	D = heuristicParameters['D'];  #6

	#remove 0-sum rows!
	sumsC = Cz.sum(axis=1);
	C=Cz[np.nonzero(sumsC>0)[0],:]

	m = max(C.shape);
	n = min(C.shape);

	# -- Prepare for decomposition
	emptyGenes,detr,translationOfRows,Cs = app.cleanInputMatrix(C,alsoReturnTranslationToOriginal=True);
	print("\t"+str(C.shape)+" -> "+str(Cs.shape))
	nodeWeights = np.zeros(max(Cs.shape));
	for x in range(m):
		iind = int(translationOfRows[x]);
		if iind >= 0:
			nodeWeights[iind] += 1;
	r = max(Cs.shape);
	#print(str(m)+" "+str(n)+" "+str(r)+" C"+str(C.sum())+" Cs"+str(Cs.sum())+"  NW"+str(nodeWeights.min())+ " sums "+str(Cs.sum(axis=0).min() )+" "+str(Cs.sum(axis=1).min() ) )
	# -- Decompose
	theBest = h1.heuristic1(Q,L,W,R,D,saveDir,savePrefix,n,m,Cs.T,numCores = nCores,numGreedy=W,geneWeights = nodeWeights,returnList=True,display=saveDir!=None)
	#print("Hfinished")

	# -- Compute variance of the sizes of each decomposition
	matrices = [Bk for Bk in theBest.values() if isinstance(Bk,DM.decomposableMatrix)]

	stds = np.zeros(m+1);
	listOfsizeMeasures = [[] for xk in range(m+1)];
	for Bk in matrices:
		MEASS = MD.sizeMeasures(Bk,C=Cs);
		stds[Bk.k] = MEASS.std;
		listOfsizeMeasures[Bk.k] = MEASS;
		if (Bk.k==r):
			Br = Bk.copy();

	#extend matrix
	#print("E")
	'''
	trans,Be = eRM.expandMatrix(Br);
	Bk = Be.copy();

	#for k > r, we just split the largest columns
	while(Bk.k < m):
		Bk.splitColumn();
		MEASS = MD.sizeMeasures(Bk);
		stds[Bk.k] = MEASS.std;
		listOfsizeMeasures[Bk.k] = MEASS;
	'''
	
	return listOfsizeMeasures,m,r

# outputData has (1+2*repetitions)*len(colVals)*len(rowVals) rows and 4 cols
# for every c,r in (colVals X rowVals) a an )m x n) matrix is created with a module of size (r x c), and repetitions random equivalents of each RAND and RSS types are generated. The matrix and all of its 2xrepetitions random equivalents are decomposed into minimum-norm decompositions of sizes k=n..m and. For each of these decompositions the standard deviation of it's module sizes is computed. Thus, at the end, for each c,r we have 1+2xrepetitions of such standard deviations. Each of them is put into a line of outputData, along with an identyfier of the type of matrix: 0: generated matrix 1:RAND equivalent 2:RSS equivalent

#moduleSizes = [[[c1,r1],[c2,r2], .... , [cl,rl]]
def experimentSeveralModules(m,n,repetitions,moduleSizes,density=0.5):
	reload(h1)
	reload(agm)

	numModules = len(moduleSizes);
	modSizesFlat = [item for sublist in moduleSizes for item in sublist]
	outputData = np.zeros([(1+2*repetitions),2*numModules+2])
	nnz = int(density*m*n);
	dataPos = 0;

	# -- Generate matrix with module
	C = createMatrix(n,m,nnz,moduleSizes);
	# -- Get the sum of variances and putit into the outputData
	sumVar,mm,rr = decomposeMatrixAndGetSumOfVariances(C)
	print([0,sumVar]+modSizesFlat)

	outputData[dataPos,[0,1]] = [0,sumVar]
	outputData[dataPos,2:]    = modSizesFlat;
	dataPos += 1;
	for rep in range(repetitions):
		C2 = createRANDEquivalent(C);
		sumVar = decomposeMatrixAndGetSumOfVariances(C2);
		print([1,sumVar]+modSizesFlat)
		outputData[dataPos,[0,1]] = [1,sumVar]
		outputData[dataPos,2:]    = modSizesFlat;
		dataPos+=1;
	for rep in range(repetitions):
		C3 = createRSS3Equivalent(C);
		sumVar = decomposeMatrixAndGetSumOfVariances(C3);
		print([2,sumVar]+modSizesFlat)
		outputData[dataPos,[0,1]] = [2,sumVar]
		outputData[dataPos,2:]    = modSizesFlat;
		dataPos+=1;

	return outputData,C,C2,C3
