'''
Author:          Victor Mireles - Freie Universitaet Berlin
Email:           syats.vm@gmail.com

Latest version:  https://github.com/syats/ModuleReusability

Distributed under GPL V3.0
'''

import aux.auxSetFunctions as sf
import gradDescent.heuristic5DMandList as h1
import numpy as np
import aux.generateSimpleMatrix as agm
import aux.auxPreprocessing as app

#typeOfSurrogate
# 1 = random matrix with same per column density
# 2 = random matrix that preserves row sum sequence in the reduced r x n matrix
# 3 = random matrix that preserves row sum sequence in the full m x n matrix

def doHeuristicAndSurrogates(origC,heuristicParameters,numSurrogates,outputDataPath,outputDataPrefix,doIntegrityTest = False,save=True,seeding=True,wVect=[],restartFromSeed1=0,typesOfSurrogates = []):
	Q = heuristicParameters['Q']
	L = heuristicParameters['L']
	W = heuristicParameters['W']
	R = heuristicParameters['R']
	D = heuristicParameters['D']
	doRealData = heuristicParameters['doRealData']
	nCores = heuristicParameters['nCores']

	if typesOfSurrogates == []:
		typesOfSurrogates = [1 for ii in range(numSurrogates)]

	if (len(wVect)>0):
		nodeWeights = np.array(wVect);
		C = origC;
		oC = origC.copy();

	else:
	#Just to remove empty genes ----
		print("cleaning");
		oC = origC.copy();
		if (origC.shape[0] > origC.shape[1]):
				oC = np.transpose(origC)


		#This removes empty tissues and duplicated genes ----
		translationOfRows,C = app.cleanInputMatrix(oC)

		m = max(C.shape);
		nodeWeights = np.zeros(m);

		''' ---  STANDARD  WEIGHTS:

		---------------------------------		'''
		for x in range(len(translationOfRows)):
			nodeWeights[translationOfRows[x]] += 1;


		''' ---  PHONY  WEIGHTS:

		---------------------------------		'''
		#nodeWeights = np.ones(m);

		''' ---- DECOMPOSABILITY MEASURE 1:
			If we make w(x) = ||C[x,:]||-1 then
			||B|| = \sum_x w(x)||B[x,:]|| and therefore
			R_b = mean_x (1 - \frac{||B[x,:]||-1}{||C[x,:]||-1})
				= 1 - ||B|| - \sum_x 1/(||C[x,:]||-1)

			R_b is the mean (over x) of a linear function that is worth 0 if ||B[x,:]||==||C[x,:]|| (that is, if every use of gene x in a condition is done via a different block), and 1 if ||B[x,::]|| == 1 (that is, if there is a single block containing x and it is used for every condition containing x).
		---------------------------------
		for x in range(m):
			if (C[x,:].sum() == 1):
				nodeWeights[x] = 1;
			else:
				nodeWeights[x] = 1.0/float(C[x,:].sum()-1);

		'''


	#Some of the methods below need an n times m matrix.
	if (C.shape[0] > C.shape[1]):
			C = np.transpose(C)

	# ------------- Compute r,n,m
	n = C.shape[0];
	m = C.shape[1];
	r = m




	if (R<=0):
		R = r;


	diffRows = set()
	for rowNum in range(m):
		row = C[:,rowNum]
		diffRows.add(tuple(row))

	print("r= "+str(len(diffRows))),


	print("n: "+str(n)+" m: "+str(m)+" sum: "+str(C.sum()));

	# This is the output data compendium. It has as many rows as possible k's, and as many coulms as surrogates + 1. The first column is for the real data, the rest are for the surrogates
	bestRealAndSurrogate = np.zeros([r+1,numSurrogates+doRealData]);

	#reload(h1)
	# ------------- First we compute for the real data
	if (doRealData == 1):
		thisPrefix = outputDataPrefix+"_REAL"
		theBest = h1.heuristic1(Q,L,W,R+2,D,outputDataPath,outputDataPrefix,n,m,C,realData = 0,numCores = nCores,numGreedy=W+1,doTest=doIntegrityTest,geneWeights = nodeWeights,display=False,printReusabilities=True,seedFromFile=seeding,restartFromSeed=restartFromSeed1)
		#print("\n\n\n\n DONE \n");
		#print(str(theBest))


	#print("Finished real data");

	# ------------- Now we compute the surrogates
	for sn in range(numSurrogates):
		typeOfSurrogate = typesOfSurrogates[sn]
		thisPrefix = outputDataPrefix+"_SU"+str(sn)
		if typeOfSurrogate==2: #RSS preserving matrix with the same n,m and r of the original matrix
			thisPrefix = outputDataPrefix+"_RSSr"+str(sn)
			transls,Cs = app.cleanInputMatrix(agm.generateRWithBinary(C.T))
			met = 1;
		if typeOfSurrogate==1: #Compliteley random matrix with the same n and m as the original matrix, and the same number of non-zero entries. It can, in general, have a different r than the original C.
			reload(agm)
			thisPrefix = outputDataPrefix+"_RAND"+str(sn)
			de = oC.sum();
			Ctemp = agm.generateRrepetitions(np.min(oC.shape),np.max(oC.shape),de);
			Mtemp = np.max(Ctemp.shape);
			transls,Cs = app.cleanInputMatrix(Ctemp)
			met = 2;
		if typeOfSurrogate==3: #RSS preserving matrix. With the original n,m and RSS. Note that this matrix's r could be different than that of the original C
			thisPrefix = outputDataPrefix+"_RSS"+str(sn)
			Ctemp = agm.generateRWithBinary(oC.T)
			Mtemp = np.max(Ctemp.shape);
			transls,Cs = app.cleanInputMatrix(Ctemp)
			met = 2;

		#transl,Cs   = app.cleanInputMatrix(Cs2);
		print("Cs _"+str(sn)+"_ m: "+str(Cs.shape[0])+" n: "+str(Cs.shape[1])+" sum: "+str(Cs.sum()));
		if save:
			np.savez_compressed(outputDataPath+thisPrefix,Cs);

		if (Cs.shape[1] < Cs.shape[0]):
			Cs = Cs.T

		if met == 1:
			nwe =  np.random.choice(nodeWeights,size=max(Cs.shape),replace=False)
		else:
			m = max(Cs.shape);
			nwe = np.zeros(m);
			for x in range(Mtemp):
				nwe[transls[x]] += 1;

		theBest = h1.heuristic1(Q,L,W,R+2,D,outputDataPath,thisPrefix,n,m,Cs,realData = 0,numCores = nCores,numGreedy=W-1,useRectangles=False,doTest=False,geneWeights = nwe,display=save)


		#print(str(Cs.sum())+" -> "+str(theBest[R]))
	# ------------- And we save the results
	if save:
		if doRealData == 1:
			label = "_COMPLETE_"
		else:
			label = "_SURROGATES_"
		np.savetxt(outputDataPath+outputDataPrefix+label+"_"+str(numSurrogates)+".csv",bestRealAndSurrogate);

	return bestRealAndSurrogate,n,r



#----------------------------------------------
#----------------------------------------------
#----------------------------------------------


def main():

	# ------------- Parameters for the overall analysis

	typesOf = [] #[3,1,3,1,3,1,3,1,3,1]
	numSurrogates = len(typesOf)

	# ------------- Paramters for the huristic
	Q = 1#4#1#3;    #The number of q's (greediness) we will try for each good matrix
	D = 1#8#0#7;   #The number of randomly-greedy propagations to try
	L = 2#10#2#7;		#The number of good matrices we will keep for every blocksize
	W = 0#90#20;   #The number of blocksizes to propagate the bad matrices
	R = -100;   #The final blocksize to look for  (if negative, it goes unto as much as possible)

	doRealData = 1;
	nCores = 4;
	wV = [];


	rFromSeed = 0;

	testPrefix = 'a1'

	# --> For tests
	'''
	Q = 2
	D = 2
	L = 3
	W = 5
	numSurrogates = 0;
	doRealData = 1;

	outputDataPath = 'smallTestsComps/';
	outputDataPrefix = 'bMPx2_';
	nCores = 3;
	R=-25;

	testPrefix = '_TEST_';
	'''
	# ------------- First load the data


	heuristicParameters = dict()
	heuristicParameters['Q'] = Q;
	heuristicParameters['L'] = L;
	heuristicParameters['W'] = W;
	heuristicParameters['R'] = R;
	heuristicParameters['D'] = D;
	heuristicParameters['doRealData'] = doRealData;
	heuristicParameters['nCores'] = nCores;


	'''
	##  --- Birds dataset ---
	basepath = '../otherArticleWork/precenseAbsense/peruBird.csv'
	origC = np.genfromtxt(basepath,delimiter='\t');
	outputDataPath = 'birdData/';
	outputDataPrefix = 'b_Btest';
	numSurrogates = 5;
	'''


	##  --- CPDB ---


	import aux.auxFunctionsCPDB as dataReader
	basepath = '../otherArticleWork/CPDB/'
	W2=dataReader.readNodeWeights(basepath);
	[nodeNames,G,nodeNumbers]=dataReader.readNetwork(basepath,W2);
	V=G.nodes();
	E=G.edges();
	#We create the graphs for the different conditions
	#origC = sf.conditionMatrixFromWeights(G,W2,nodeNames)
	origC,gnames = dataReader.readWholeMatrix(basepath);
	origC = origC.T;

	outputDataPath = 'cpdbReuse/';
	outputDataPrefix = 'Re'+testPrefix;
	print("LOADED::  "+str(origC.shape))


	## --- BRAWAND 2011 --------
	'''
	fileName = '../otherArticleWork/Brawand2011/humanBinaryClean.npz';
	AA= np.load(open(fileName,'rb'));
	origC = AA['c'];
	wv = AA['wV']

	wV = wv.tolist()

	outputDataPath = 'brawandData/';
	outputDataPrefix = 'Hq1_1010_'+testPrefix;

	rFromSeed = 23462;

	'''




	bestRealAndSurrogate,n,r = doHeuristicAndSurrogates(origC,heuristicParameters,numSurrogates,outputDataPath,outputDataPrefix,doIntegrityTest = False,seeding=False,wVect=wV,restartFromSeed1=rFromSeed,typesOfSurrogates=typesOf);
	print(str(origC.shape)+" "+str(origC.sum()))

	return bestRealAndSurrogate
