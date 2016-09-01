'''
Author:          Victor Mireles - Freie Universitaet Berlin
Email:           syats.vm@gmail.com

Latest version:  https://github.com/syats/ModuleReusability

Distributed under GPL V3.0
'''


import numpy as np
import pickle
import gc
import datetime
import random as ran
import auxForMultiCore as amp
import aux.auxGeneratingModules as agm
import os.path
import copy

import gradDescent.decomposableMatrix as DM
import gradDescent.listOfDecompositions as LOD



#Q = 4;    #The number of q's (greediness) we will try for each good matrix
#l = 8; 	 #The number of good matrices we will keep for every blocksize
#w = 16;   #The number of blocksizes to propagate the bad matrices
#r = 820;  #The final blocksize to look for
#D = Q;    #The number of randomly-greedy propagations to try
def heuristic1(Q,l,w,r,D,outputDataPathO,outputDataPrefixO,n,m,C,realData = 0,numCores = 1,numGreedy=2,useRectangles=False,doTest = False,geneWeights = [],display=False,seedFromFile=False,restartFromSeed=0,printReusabilities=False,returnList=False,removeSuperflousFromB=True):

	reload(DM)
	reload(LOD)

	CDM = DM.decomposableMatrix(C,wVector = np.array(geneWeights));


	m   =   CDM.m;  # number of genes
	n   =   CDM.k;  # numbre of tissues
	R   =   min([r,m]);              # number of propagations to try
	if (R < 0):
		R = m+l;

	CC = C;
	if (C.shape[0]==n):
		CC = C.T;

	if outputDataPathO == None:
		outputDataPath = '/dev/null/'
	else:
		outputDataPath =outputDataPathO
	if outputDataPrefixO == None:
		outputDataPrefix = ''
	else:
		outputDataPrefix = outputDataPrefixO

	# In outFile we save the best norm for every k
	fileName = outputDataPath+outputDataPrefix+"_BestScores_"+str(Q)+"_"+str(l)+"_"+str(w)+".dat";
	if (display):
		outFile = open(fileName, 'w')

	reload(LOD)
	reload(DM)
	#L = listOfDecompositions(C,numCores);
	nGreedy = numGreedy;
	if (restartFromSeed > 0):
		nGreedy = 0;
	L = LOD.listOfDecompositions(CDM,l,D,w,nGreedy,realData,r,numCores)
	#L.init(C,numGreedy,blabla)
	if (restartFromSeed>0) or (seedFromFile):
		L.seed(outputDataPath,outputDataPrefix,C,restartFromSeed)
	counter = 0;
	while (L.smallestNonEmpty() <= m) or (L.bestNorms[m] > m):

		K = L.smallestNonEmpty()

		#print(str(K)+" "+str(len(L.theList[K]))),

		if (counter >= l):
			break;

		if ((K==m) and (L.bestNorms[m]==m)):
			counter += 1;
		if len(L.theList[K]) == 0:
			break


		oldBs = [];
		#Here we remove superflous 1's in the best for this K
		supTaken = 0;
		if removeSuperflousFromB:
			newBs = []
			for bkWS in L.theList[K]:
				if supTaken > 1:
					break
				bkWSC = bkWS.copy();
				newBk = bkWSC.removeSuperflousIncreasingReuse(CC)
				if newBk != None:
					newBs.append(newBk)
					#print("\n added "+str(newBk.weightVector.sum()))
					supTaken+=1;
					if newBk.k == K:
						oldBs.append(bkWSC)
			L.addMatrices(newBs)


		#print(" -> "+str(len(L.theList[K]))+ " coi: "+str(L.theList[K][0].theCoincidM.sum())+" sha"+str(L.theList[K][0].theMatrix.shape)+" nor"+str(L.theList[K][0].norm()))
		toPropagate  =  L.theList[K];
		L.bestNorms[K] = L.theList[K][0].norm0();

		#These we need for saving and output
		bestThisK      = L.theList[K][0];

		if (returnList):
			L.bestDecomps[K] = bestThisK.copy();

		best0Normp     = L.bestNorms[K]
		best0Norm      = min([BB.norm0() for BB in L.theList[K]])
		L.bestNorms[K] = best0Norm;
		forK           = list(L.theList[K])

		#We clear this K
		L.theList[K]   = [];
		L.ocupations[K] = 0;

		propagated   =  L.propagateOneStep(toPropagate,Q,D,l*(Q+D))
		L.addMatrices(propagated)

		Ww = w;
		Ll = l
		#If we can't diminish ||Bk||_0 by much, we make W=0
		if ( bestThisK.theMatrix.indices.shape[0] - R ) <= 2 * ( R - K ):
			Ww = 0;
			Ll = -1;

		lim1 = min( [ K + l  , R+1 , len(L.theList)] );
		lim2 = min( [ lim1  + Ll+ran.choice(range(Q))  , R+1 , len(L.theList) ] )
		lim3 = min( [ lim2  + Ww , R+1 , len(L.theList) ] )


		range1 = range(K+1,lim1)
		range2 = range(lim1,lim2)
		range3 = range(lim2,lim3)

		#print(" ::1:-> "),
		for k2 in range1:
			dd = 0;
			if (ran.random() < 0.10):
				dd = 1;
			#we leave l and propagate the rest
			toPropagate   = L.theList[k2][-(l*(Q+D)):];
			if (len(toPropagate) < l*(Q+D)) and (len(L.theList[k2])>1):
				missing = l*(Q+D) - len(toPropagate);
				toPropagate.extend([ran.choice(L.theList[k2][:l]) for kc in range(missing+1)  ])

			if (K>=n+l+1+numGreedy+dd):
				L.trim(k2-1);

			ll = max(len(L.theList[k2])-l,l);
			if ll < 2:
				ll = 2;

			[good, bad]   = L.chooseLbest(L.theList[k2],ll);
			if any([B.k != k2 for B in good  ]  ):
				L.addMatrices(good)
			else:
				L.theList[k2] = good
			L.ocupations[k2] = len(L.theList[k2]);
			propagated    = L.propagateOneStep(toPropagate,1,dd,len(toPropagate)+dd*l);

			L.addMatrices(propagated)
			if (len(L.theList[k2]) > 0):
				L.bestNorms[k2] = L.theList[k2][0].norm0();

		if (L.smallestNonEmpty() < K):
			continue;


		toPropagate   = propagated[l:2*l];
		if (len(toPropagate) < l) and (len(propagated)>0):
			missing = l - len(toPropagate);
			toPropagate.extend([ran.choice(propagated[:l]) for kc in range(missing)  ])
		#print("  :2 ( "+str(len(L.theList[range2[0]]))+" )->"),
		for k2 in range2:
			dd = 0;
			if (ran.random() < 0.075):
				dd = 1;

			#we leave l and propagate the rest
			#toPropagate   = L.theList[k2][l:2*l];
			#print(str(k2)+"("+str(len(toPropagate))+") "),
			ll = l-supTaken;
			if ll < 2:
				ll = 2;

			[good, bad]   = L.chooseLbest(L.theList[k2],ll);
			if any([B.k != k2 for B in good  ]  ):
				#print("You are putting good in the wrong place" );
				L.addMatrices(good)
			else:
				L.theList[k2] = good
			L.ocupations[k2] = len(L.theList[k2]);
			propagated    = L.propagateOneStep(toPropagate,1,dd,len(toPropagate)+dd);
			L.addMatrices(propagated)
			if (k2 > n+numGreedy+dd+1) and (k2-K)>4:
				L.trim(k2-1)

			toPropagate = propagated;


		if (L.smallestNonEmpty() < K):
			continue;


		#print("  :3 ( "+str(len(L.theList[range3[0]]))+" )->"),
		for k2 in range3:
			dd = 0;
			if (ran.random() < 0.07):
				dd = 1;
			#print(str(k2)+"("+str(len(toPropagate))+") "),
			propagated = L.propagateOneStep(toPropagate,1,dd,numCores+dd,alsoBad=(k2-lim2 <= Q+dd))
			#print("("+str(len(toPropagate))+") "),
			if len(propagated)==0:
				break
			si = propagated[0].k

			#print(propagated)
			L.theList[si].extend(propagated);
			#L.addMatrices(propagated)
			L.ocupations[si] = len(L.theList[si])
			if (k2 > n+numGreedy+1+dd):
				L.trim(k2-1)
			toPropagate = propagated;

		if (L.smallestNonEmpty() < K):
			continue;
		#print("  ")
		#test
		#display
		#printResusabilities
		#print(str(K)+": listLen "+str(len(L.theList[K]))+ " ocup"+str(L.ocupations[K]));


		L.restartMulticore();
		gc.collect();
		#print([tuple([X,L.ocupations[X],len(L.theList[X]) ]) for X in np.nonzero(L.ocupations)[0].tolist()])
		#print("-");


		if (doTest):
			tDense = bestThisK.getMatrix().toarray()
			#print(str(tDense.T.shape)+" "+str(C.shape))
			ErX = testBonC(tDense.T,CC.T  )
			if True:
				print(str(tDense.shape)),
				print("E>:"+str( ErX)+ " ("+str(len(np.nonzero(tDense.sum(axis=0)==0)[0]))+"-"+str(len(np.nonzero(tDense.sum(axis=1)==0)[0]))+")"),

		if (display):
			print("K:"+str(K)+" "+str(len(forK))+" norm: "+str(best0Norm)+" ("+str(bestThisK.getMatrix().indices.shape[0])+") shape: "+str([bestThisK.m , bestThisK.k])+" "+str(datetime.datetime.now()))
			#Matrix save
			matrixFileName = outputDataPath+outputDataPrefix+"_Matrix_"+str(bestThisK.k)+".decM";
			bestThisK.sortIndicesWithinColumns();
			#np.savez_compressed(matrixFileName,theBest[k][0].theMatrix);
			bestThisK.saveToFile(matrixFileName);
			#Score save
			#outFile.write(str(bestThisK.k)+" \t "+ str(bestThisK.norm())+"\n")
			#outFile.flush()

		if (printReusabilities):
			for bb in forK:
				print(str(K)+" "+str(bb.norm())+" "+str(bb.reusability(CC,normalized=True)));

		del forK

	if (display):
		print(str([tuple([K,L.bestNorms[K]]) for K in range(n,m+1)]))
		for K in range(n,m+1):
			outFile.write(str(K)+" \t "+ str(L.bestNorms[K])+"\n")

		outFile.flush()

	L.multiCoreHelper.kill();

	if (returnList):
		return L.bestDecomps
	else:
		return [L.bestNorms[K] for K in range(R+1)  ]




def testBonC(BkO,CO):
	Omega = np.dot(CO,BkO.T);
	SkO = np.zeros([CO.shape[0], BkO.shape[0]]);
	#Omega[a,b] = number of genes both in block b and tissue a
	for inB in range(SkO.shape[1]):
		sizB = BkO[inB,:].sum();
		for inT in range(SkO.shape[0]):
			if (Omega[inT,inB] == sizB):
				SkO[inT,inB] = 1;
	return agm.testBk(BkO,SkO,CO)
