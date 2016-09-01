'''
Author:          Victor Mireles - Freie Universitaet Berlin
Email:           syats.vm@gmail.com

Latest version:  https://github.com/syats/ModuleReusability

Distributed under GPL V3.0
'''


import decomposableMatrix as DM
import numpy as np
import random as ran
import math
from scipy.stats import binom
from scipy.stats import entropy
from scipy.special import comb
import bisect
from scipy.stats import entropy
from scipy.stats import histogram

class sizeMeasures:

	def __init__(self,B,C=None,modSizes=[],Qpos=[0.05,0.95],numBins=-1):
		if C==None:
			rsizes = allModuleSizes(B)
		else:
			perModReuse,perModIntersect,Rsizes = perModuleReusability(B,C,alsoWeights=True)
			rsizes = np.array(Rsizes)

		if len(modSizes)==0:
			sizes = rsizes;
		else:
			sizes = modSizes;

		realM = B.weightVector.sum()
		if realM < 2*numBins:
			print("*")
			numBins = int(realM/2);
		if numBins < 0:
			numBins = int(realM/float(2))
		hist  = histogram(sizes,numbins=numBins,defaultlimits=(1,realM))[0]
		K = len(sizes)

		if not (C==None):
			sizeReuseParis = [(sizes[j],perModReuse[j]) for j in range(K)]
			self.SRpairs    = sizeReuseParis;
		else:
			self.SRpairs 	= [];

		sizes.sort();

		self.allSizes   = sizes;
		self.histogram  = hist;
		self.k 			= K;
		self.median 	= sizes[int((K-1)/2)]
		self.mean		= np.mean(sizes)
		self.maxi		= sizes[-1]
		self.mini		= sizes[0];
		self.std		= np.std(sizes);
		self.H			= entropy(hist,base=2)+0;
		self.Qs			= np.zeros(len(Qpos));
		self.total		= sizes.sum()
		self.numUnique	= np.unique(sizes).shape[0]

		for qpn in range(len(Qpos)):
			qp = Qpos[qpn];
			self.Qs[qpn] = sizes[int((K-1)*qp)]



def allModuleSizes(B):
	return np.array([B.realSizes(j) for j in range(B.k)])

def perModuleExpectedReusabilityRSS(B,C):
	perModExp = np.zeros(B.k);
	wV = B.weightVector;
	n = min(C.shape)
	ra = np.array(range(n+1))
	for j in range(B.k):
		ijstart = B.theMatrix.indptr[j];
		ijend = B.theMatrix.indptr[j+1];
		indj = B.theMatrix.indices[ijstart:ijend];
		exj = 0;
		#In how many tissues is each gene contained
		N  = C[indj,:].sum(axis=1);

		den = 1;
		for x in range(len(indj)):
			nx = N[x];
			den *= comb(n,nx)

		for k in range(1,N.min()+1):  #SUM -------
			pos = 1;
			neg = 1;
			if (N.min() < k+1) or (k==n):
				neg = 0;
			for x in range(len(indj)): #PRODUCT ------
				nx = N[x];
				pos *= comb(n-k,nx-k);
				if neg > 0:
					neg *= comb(n-k-1,nx-k-1)

			sumand = k*(comb(n,k)*pos - comb(n,k+1)*neg);
			if (comb(n,k)*pos)/den > 1:
				print(str(den)+" ."),
			exj += sumand;



		perModExp[j] = exj/den;
	return perModExp

#For every colum b of thematrix, returns two numbers: it's reuse: in how many columns of C does it "fit completely" (all non-zero rows of b are non zero rows in the column of C); and it's intersection: how many colums of C does it "intersect" (at least one nonzero row of b is a nonzero row in the column of C)
def perModuleReusability(B,C,alsoWeights=False):
	reus = 0;
	totLik = 0;
	perModReuse = np.zeros(B.k);
	perModIntersect = np.zeros(B.k);
	rsizes = np.zeros(B.k);


	N = np.min(C.shape);
	for j in range(B.k):
		ijstart = B.theMatrix.indptr[j];
		ijend = B.theMatrix.indptr[j+1];
		indj = B.theMatrix.indices[ijstart:ijend];
		sizeB = B.realSizes(j);
		ri = 0;
		lindj = len(indj);

		#for every tissue, which of the genes of j it cotains
		Cj = C[indj,:];
		sCj = Cj.sum(axis=0);
		sCj = np.squeeze(np.asarray(sCj))
		#tissues that contain at least one of the genes in bj
		tj = len(np.nonzero(sCj>0)[0]);
		ri = len(np.nonzero(sCj==lindj)[0]);
		if (ri==0):
			print("\nZERO REUSE!!! "+str(B.k)+" "+str(j)+" "+str(lindj))
			print(indj)

		perModReuse[j]=ri;
		perModIntersect[j] = tj;
		rsizes[j] = sizeB;

	minReuse = min(perModReuse)


	if alsoWeights:
		return perModReuse,perModIntersect,rsizes;
	else:
		return perModReuse,perModIntersect
