'''
Author:          Victor Mireles - Freie Universitaet Berlin
Email:           syats.vm@gmail.com

Latest version:  https://github.com/syats/ModuleReusability

Distributed under GPL V3.0
'''


import numpy as np
import pickle
import glob
import scipy as sci
from scipy.stats import entropy

import permutationsOfArtificialModules as pam
import matplotlib.pyplot as plt
import plottingWithRandEquivalents as  pt
from   gradDescent.measuresForDecomposableMatrix import sizeMeasures


#For a given real mxn matrix and set of surrogates of it, it stores a set of curves (scalar values associated to the decompositions of a matrix into k blocks for k in [n,m]) and, optionally, the decompositions of it. The surrogates are matrices that are in some sense equivalent to the real matrix, and can be of different types. This class supports writing and reading this data onto disk.
#realC is an mxn matrix.  realCurves is a dictionary, whose keys are the names of curves (for example 'StdsOf0Norm' or 'SumsOf0Norm') and whose items are lists of m elements.
#surrogateCs[i] is an mxn matrix. surrogateTypes[i] is a string (eg. 'REAL', 'RAND' of 'RSS3'). surrogateCurves[i] is a dict whose keys are the names of curves (for example 'StdsOf0Norm' or 'SumsOf0Norm') and whose items are lists of m elements.

class decompositionWithSurrogates(object):
	""",  Stores information on a decomposition of a real matrix and some random equivalents of it (surrogates). The main aim of this is to refrain from recompouting decompositions and surrogates which  might take a while"""
	def __init__(self, description, Creal=None,heuristicParameters=None):
		self.realC = Creal;
		self.realHash = hash(str(Creal));
		self.realCurves = [];
		self.realDecompositions = [];

		self.surrogateCs = [];
		self.surrogateTypes = [];
		self.surrogateCurves = [];
		self.surrogateDecompositions = []
		self.surrogateHashes = [];

		self.description = description;
		self.heuristicParameters = heuristicParameters;

	def addC(self,C,typeC,curves,curveNames,Bs=[]):
		if typeC=='real' or typeC=='REAL':
			self.realC = C;
			d = dict();
			for cn in range(len(curveNames)):
				name 	= curveNames[cn];
				d[name] = curves[cn];
			self.realCurves = d;
			self.realDecompositions = Bs;
			self.realHash = hash(str(C.tolist()));

		else:
			self.surrogateCs.append(C);
			self.surrogateTypes.append(typeC)
			self.surrogateDecompositions.append(Bs);
			self.surrogateHashes.append(hash(str(C)));
			d = dict();
			for cn in range(len(curveNames)):
				name 	= curveNames[cn];
				d[name] = curves[cn];
			self.surrogateCurves.append(d);

	def saveToFile(self,directory,smallDesc=""):
		if directory[-1] != '/':
			directory = directory + "/";
		smallDescription = "".join([c for c in smallDesc if c.isalpha() or c.isdigit() or c==' ']).rstrip()
		totHash = str(self.realHash)+"_"+str(self.realC.shape[0])+"."+str(self.realC.shape[1])+"_"+str(self.realC.sum());
		fileName = directory+smallDescription+"_H"+totHash+".dws";
		fileName = fileName.replace(" ","_");
		pickle.dump(self,open(fileName,"wb"));
		return totHash



def myH(alist):
	if len(alist)==0:
		return np.nan
	blist = np.array(alist)
	bmax = int(blist.max())
	theHist = np.zeros(bmax+1);
	for rv in range(1,bmax+1):
		theHist[rv] = len(np.nonzero(blist==rv)[0])
	return entropy(theHist)


def flattenSizeReuseList(allData,origCs,sizeOrReuse,MeanMaxOrEnt):
	compactAllData = dict();
	for ke in allData.keys():
		C = origCs[ke];

		m = max(C.shape);
		compactAllData[ke] = []
		smallestNan = np.inf;
		for dataSet in allData[ke]:


			if MeanMaxOrEnt == 0:
				newDataSet = np.array([np.mean([x[sizeOrReuse] for x in dp]) if len(dp)>0 else np.nan   for dp in dataSet]);
			if MeanMaxOrEnt == 1:
				newDataSet = np.array([np.max([x[sizeOrReuse] for x in dp]) if len(dp)>0 else np.nan   for dp in dataSet]);
			if MeanMaxOrEnt == 2:
				newDataSet = np.array([myH([x[sizeOrReuse] for x in dp]) if len(dp)>0 else np.nan   for dp in dataSet]);


			nans = np.nonzero(np.isnan(newDataSet)==True)[0];
			lasNonNan = nans[0]-1;
			if lasNonNan < smallestNan:
				#print("NANupdate")
				smallestNan = lasNonNan
			compactAllData[ke].append(newDataSet)

		for dnum in range(len(compactAllData[ke])):
			compactAllData[ke][dnum] = compactAllData[ke][dnum][:smallestNan];

	return compactAllData





def loadOrCompute(fn,C,storageDir,
					totHash,
					heuristicParms,
					functionsToCompute,
					thisNumRSS3O=2,
					thisNumRandO=2,
					nc=1,
					saveDir=None, savePrefix=None
					):
	storedData = loadAccordingToRealCHash(storageDir,totHash);
	dataThisFn = []
	typesThisFn = []
	#We first check what we have saved for this matrix
	thisNumRand = thisNumRandO;
	thisNumRSS3 = thisNumRSS3O;
	if  isinstance(storedData,decompositionWithSurrogates) and (functionsToCompute[0] in storedData.realCurves.keys()):
		stds = storedData.realCurves[functionsToCompute[0]];
		dataThisFn.append(stds)
		typesThisFn.append(0)

		print("stypes: "+str(storedData.surrogateTypes))

		numSugs = len(storedData.surrogateHashes);
		for ns in range(numSugs):
			typeOfS = storedData.surrogateTypes[ns];
			numericalType = -1
			if typeOfS == 'RSS3' and thisNumRSS3>0:
				numericalType = 2;
				thisNumRSS3 -= 1;
			if typeOfS == 'RAND' and thisNumRand>0:
				numericalType = 1;
				thisNumRand -= 1;
			if numericalType == -1:
				continue
			stds    = storedData.surrogateCurves[ns][functionsToCompute[0]];
			dataThisFn.append(stds)
			typesThisFn.append(numericalType);
			#print(str(C.shape)+" "+str(stds.shape))

		print("loaded! "+str(thisNumRand)+","+str(thisNumRSS3))
		toStore = storedData;

	else:
		toStore = decompositionWithSurrogates(fn);
		print("COMPUTING: "+fn)
		allSizesMeasuresForThisK,m,r = pam.decomposeMatrixAndGetSumOfVariances(C,nCores=nc,heuristicParameters=heuristicParms,returnlistOfsizeMeasures=True,saveDir=saveDir,savePrefix=savePrefix);
		curvesToStore = []
		#allSizesMeasuresForThisK = [sdata for sdata in allSizesMeasuresForThisK if len(sdata) > 0]
		print("0: "+str(C.shape)+" m"+str(m))
		#print([ [MM.k , len(MM.histogram)] for MM in  allSizesMeasuresForThisK if isinstance(MM,sizeMeasures) ])
		for fun in functionsToCompute:
			toAppend = [MM.__dict__[fun] for MM in  allSizesMeasuresForThisK if isinstance(MM,sizeMeasures) ]
			lta = len(toAppend)
			if (fun=='histogram') or (fun =='SRpairs'):
				curvesToStore.append(toAppend)
			else:
				toAppend = [0 for xx in range(m+1-lta)] + toAppend;
				curvesToStore.append(np.array(toAppend));
		toStore.addC(C,'REAL',curvesToStore,functionsToCompute)
		dataThisFn.append(curvesToStore[0])
		typesThisFn.append(0);

		toStore.saveToFile(storageDir)




	#Whether the file was found for loading or not, we might be missing some surrogates
	if thisNumRand < 0:
		thisNumRand = 0;
	if thisNumRSS3 < 0:
		thisNumRSS3 = 0;

	for rn in range(thisNumRand):
		CR = pam.createRANDEquivalent(C)
		allSizesMeasuresForThisK,m,r = pam.decomposeMatrixAndGetSumOfVariances(CR,nCores=nc,heuristicParameters=heuristicParms,returnlistOfsizeMeasures=True);
		curvesToStore = []
		#allSizesMeasuresForThisK = [sdata for sdata in allSizesMeasuresForThisK if len(sdata) > 0]
		for fun in functionsToCompute:
			toAppend = [MM.__dict__[fun] for MM in  allSizesMeasuresForThisK if isinstance(MM,sizeMeasures) ]
			if (fun=='histogram') or (fun =='SRpairs'):
				curvesToStore.append(toAppend)
			else:
				toAppend = [0 for xx in range(m+1-len(toAppend))] + toAppend;
				curvesToStore.append(np.array(toAppend));
		toStore.addC(CR,'RAND',curvesToStore,functionsToCompute)
		dataThisFn.append(curvesToStore[0])
		typesThisFn.append(1);
		print('RAND')

	for rn in range(thisNumRSS3):
		CR = pam.createRSS3Equivalent(C)
		allSizesMeasuresForThisK,m,r = pam.decomposeMatrixAndGetSumOfVariances(CR,nCores=nc,heuristicParameters=heuristicParms,returnlistOfsizeMeasures=True);
		curvesToStore = []
		#allSizesMeasuresForThisK = [sdata for sdata in allSizesMeasuresForThisK if len(sdata) > 0]
		for fun in functionsToCompute:
			toAppend = [MM.__dict__[fun] for MM in  allSizesMeasuresForThisK if isinstance(MM,sizeMeasures) ]
			if (fun=='histogram') or (fun =='SRpairs'):
				curvesToStore.append(toAppend)
			else:
				toAppend = [0 for xx in range(m+1-len(toAppend))] + toAppend;
				curvesToStore.append(np.array(toAppend));
		toStore.addC(CR,'RSS3',curvesToStore,functionsToCompute)
		dataThisFn.append(curvesToStore[0])
		typesThisFn.append(2);
		print('RSS')

	if thisNumRSS3 + thisNumRand > 0:
		toStore.saveToFile(storageDir)

	return dataThisFn,typesThisFn,toStore



def loadAccordingToRealCHash(directory,hast,hparameters=[]):
	print(hast)
	if directory[-1] != '/':
		directory = directory + "/";
	possibleFiles = glob.glob(directory+"*_H"+str(hast)+".dws");
	print(possibleFiles)
	if len(possibleFiles)<1:
		return 0

	#If there's only one file and no specific parameters were requested
	if len(hparameters)==0 and len(possibleFiles)==1:
		return pickle.load(open(possibleFiles[0],"rb"))

	#If no specific parameters were requested but there are multiple files
	if len(hparameters)==0 and len(possibleFiles)>1:
		return len(possibleFiles)

	#If specific parameters where requested, we go through all files. We discard those that have no parameters stored, or those that lack one of the parameters requested. If we find a file that matches all of the requested parameters we return it
	numFoundCandidates = 0;
	lastGoodCandidate  = None;
	for posFile in possibleFiles:
		posDWS = pickle.load(open(posFile,"rb"))
		if posDWS.heuristicParameters == None:
			continue
		if not(all([x in posDWS.heuristicParameters.keys() for x in hparameters.keys() ])):
			continue
		if all([posDWS.heuristicParameters[x] == hparameters[x] for x in hparameters.keys()   ] ):
			lastGoodCandidate = posDWS;
			numFoundCandidates += 1;

	#If we found exactly one dws matching the requested parameters
	if (numFoundCandidates == 1):
		return lastGoodCandidate;

	return numFoundCandidates;
