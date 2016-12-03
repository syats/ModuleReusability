'''
Author:          Victor Mireles - Freie Universitaet Berlin
Email:           syats.vm@gmail.com

Latest version:  https://github.com/syats/ModuleReusability

Distributed under GPL V3.0
'''


import numpy as np
import random as ran
import os.path
import gc;

import aux.auxGeneratingModules as agm
import auxForMultiCore as amp
import decomposableMatrix as DM



class listOfDecompositions:

	'''
	INIT                  -------------
	'''
	# theList{k} contains a list of decomposableMatrix objects, sorted by norm.
	def __init__(self,CDM,l,D,w,numGreedy,realData,R,numCores = 1):

		#CDM = DM.decomposableMatrix(C,wVector = np.array(geneWeights));
		m = CDM.m;
		n = CDM.k;
		self.minSize     = n;
		self.maxSize     = 2*m+n;
		self.theList     = {x:[] for x in range(self.maxSize)};
		self.bestDecomps = {x:[] for x in range(self.maxSize)};
		self.bestNorms  = [np.Inf for x in range(self.maxSize)];
		self.ocupations = np.zeros(self.maxSize);
		self.numCores   = numCores;
		self.l          = l;
		self.smallestEver = [np.Inf for x in range(self.maxSize)];


		#print(str(numCores))
		self.multiCoreHelper = amp.auxForMultiCore(numCores)

		''' ------------------- List initialization
		'''

		Q = 3 + D;
		theBest = [ [] for k in range(R+1+w) ];


		#We intialize the list of bests for k = n
		self.theList[n]    = [CDM.copy()  for c in range(int(l/2)+1)] ;
		self.ocupations[n] = l;
		toProp  = [a.copy() for a in self.theList[n]];
		#And then fill up the list of bests for the next w k's
		Bk = CDM.copy();

		#print("startInit: "+str(Bk.theMatrix.shape))
		for k in range(n+1,n+numGreedy+2*l):
			Bk.alsoSmaller = True;
			Bk.mergeOverlappingColumns();
			#print(str(k)+" ->-> "+str(Bk.theMatrix.shape)+" toprop:"+str(len(toProp)));
			toProp2 = self.propagateOneStep(toProp,Q+2,D,l*(Q+D)+realData*D)
			size = len(toProp2);
			[good , bad] = self.chooseLbest(toProp2,size);
			#print(str(len(good))+" "+str(len(bad)))
			self.addMatrices(good+[Bk.copy()])
			L = l
			if len(good) < l:
				L = len(good)
			toProp = [a.copy() for a in good[:L]]+[Bk.copy()];
			#print(str(k)+" prop="+str(len(toProp))+" stay="+str(len(self.theList[k])))


	def seed(self,outputDataPath,outputDataPrefix,C,restartFromSeed=0):
		n = min(C.shape);
		m = max(C.shape);
		count = 0;
		toAdd = [];
		start = max([n+1,restartFromSeed-1]);
		print("seedingStart= "+str(start));
		for k in range(start,m):
			fileName = outputDataPath+outputDataPrefix+"_Matrix_"+str(k)+".decM.npz"
			if (os.path.isfile(fileName)):
				a  = DM.decomposableMatrix()
				Bk = a.loadFromFile(fileName);
				if (restartFromSeed>0):
					for k2 in range(n,k):
						self.theList[k2] = [];
					self.ocupations[range(n,k)] = 0;
					toAdd = [Bk];
					count = 1;

				else:
					tDense = Bk.getMatrix().toarray()
					ErX = self.testBonC(tDense.T,C)
					if (ErX == 0) and (Bk.k == k) and (Bk.m == m):
						toAdd.append(Bk);
						count += 1;
			else:
				break

		self.addMatrices(toAdd);
		print("seeded: "+str(count));

	def testBonC(self,BkO,CO):
		Omega = np.dot(CO,BkO.T);
		SkO = np.zeros([CO.shape[0], BkO.shape[0]]);
		#Omega[a,b] = number of genes both in block b and tissue a
		for inB in range(SkO.shape[1]):
			sizB = BkO[inB,:].sum();
			for inT in range(SkO.shape[0]):
				if (Omega[inT,inB] == sizB):
					SkO[inT,inB] = 1;
		return agm.testBk(BkO,SkO,CO)


	'''
	mapPropagations           -------------
	'''
	def mapPropagations(self,toProp):
		multiCoreHelper = self.multiCoreHelper;
		if (multiCoreHelper.nCores == 1) or (len(toProp) < 3):
			mapResult  =  map(amp.singlePropagationDM,toProp)
		else:
			mapResult  =  multiCoreHelper.propagateMultiCoreDM(toProp);

		# mapResult if a list of lists of tuples
		# this should return a list of tuples
		return [ x for y in mapResult for x in y ]


	'''
	propagateOneStep       -------------
	'''
	def propagateOneStep(self,matsO,Q,D,LL,maxNorm=np.inf,alsoBad=False):

		mats = [bb for bb in matsO if bb.norm0() > min(bb.theMatrix.shape)];
		L = LL-(len(matsO)-len(mats));
		if (L<1):
			L = 1;

		if (len(mats) < 1):
			return []

		k = mats[0].k;
		toProp = [];
		#The greedy propagations
		for B in mats:
			B.alsoSmaller = True;
			toProp.extend([tuple([B,q]) for q in range(Q)])

		allPropagatedTriplets = self.mapPropagations(toProp);
		#print("first greedy "+str(len(allPropagatedTriplets)))

		#The repetitions
		if (D>0):
			toProp = [tuple([t[3],t[2]]) for t in allPropagatedTriplets if (t[1] > 1)];
		else:
			toProp = [tuple([t[3],t[2]]) for t in allPropagatedTriplets if (t[1] > 1) and (ran.random() < 0.050) ];

		#The random ones
		for B in mats:
			tP = [tuple([B,Q+ran.choice(range(2+k))]) for d in range(D)]
			toProp.extend(tP);

		propTriplets = self.mapPropagations(toProp)
		del(toProp)
		allPropagatedTriplets.extend(propTriplets);

		#print("after random and reps"+str(len(allPropagatedTriplets)))

		#Small cleaning
		prop = [t[0] for t in allPropagatedTriplets if ( (t[0].k >= k+1) and  (t[0].norm() < t[3].norm()) )];
		propSmall = [t[0] for t in allPropagatedTriplets if ( (t[0].k < k+1) and  (t[0].norm() < t[3].norm()) )]

		#Remove large ones
		prop = [pp for pp in prop if pp.norm() <= maxNorm]

		#Remove repeated ones
		sofar = len(prop)
		if (sofar > 0):
			[prop, EMPT] = self.chooseLbest(prop,L);
			#print("after removing repeats "+str(len(prop))+" "+str(len(EMPT)))
			del EMPT
			sofar = len([x for x in prop if x.k == k+1]);

		#If after removing repeated we are missing some, we add some random propagations
		attemps = 10*(L+Q+D)
		while ( ((L-sofar) > 0)  and (len(mats) > 0) ) and (attemps > 0):
			attemps -= 1;
			toProp = [tuple([ran.choice(mats),ran.choice(range(1,2*k))]) for d in range(L-sofar+1)]
			propTriplets = self.mapPropagations(toProp)
			prop2 = [t[0] for t in propTriplets if (t[0].k >= k+1) and (t[0].norm() < t[3].norm())];
			propSmall2 = [t[0] for t in allPropagatedTriplets if ( (t[0].k < k+1) and  (t[0].norm() < t[3].norm()) )]
			prop2 = [pp for pp in prop2 if pp.norm() <= maxNorm]
			prop.extend(prop2);
			propSmall.extend(propSmall2);
			sofar = len([x for x in prop if x.k == k+1]);
			if (k+1 >= 0.85*mats[0].m):
				sofar = L;

		if (len(prop) > L):
			[prop, EMPT] = self.chooseLbest(prop,L);

		#propSmall, EMPT = self.removeRepeated(propSmall)
		#prop, EMPT = self.removeRepeated(prop)

		badProp  = []
		if (alsoBad):
			toProp = [];
			for B in mats:
				B.alsoSmaller = True;
				posOv = len(B.getListOfOverlaps());
				if (posOv <= Q):
					continue
				#print(str(ovToUse)+" "),
				toProp.append(tuple([B,posOv-1]))
			allPropagatedTriplets = self.mapPropagations(toProp);
			badProp = [ t[0] for t in allPropagatedTriplets if (t[0].norm() < t[3].norm()) ]

		finalList = prop+propSmall+badProp
		if (k==-17):
			print("17>>>> "+str([x.theMatrix.shape for x in finalList]))
		return finalList

	'''
	removeRepeated           -------------
	'''
	def removeRepeated(self,listOfMatricesO):
		if len(listOfMatricesO)==0:
			return [],[]
		if listOfMatricesO[0].C == None:
			norms = np.array([pp.norm() for pp in listOfMatricesO])
		else:
			norms = np.array([1.0/pp.reusability(pp.C) for pp in listOfMatricesO])
		onorms = np.argsort(norms);
		norms = [norms[i] for i in onorms.tolist()];
		listOfMatrices = [listOfMatricesO[i] for i in onorms.tolist()  ];

		tokeep = np.ones(len(listOfMatricesO));
		for l in range(1,len(listOfMatrices)):
			if (norms[l-1] == norms[l]):
				if (listOfMatrices[l-1].equals(listOfMatrices[l])):
					tokeep[l] = 0;

		toKeepI = np.nonzero(tokeep)[0];
		listOfMatrices = [listOfMatrices[i] for i in toKeepI];
		Norms          = [norms[i] for i in toKeepI];

		return listOfMatrices,Norms


	'''
	chooseLbest          -------------
	'''
	def chooseLbest(self,listOfMatricesO,l):

		listOfmatrices,norms = self.removeRepeated(listOfMatricesO);

		I = list(np.argsort(norms));

		l2 = l
		if l > len(I):
			l2 = len(I)
		good = [listOfmatrices[x] for x in I[:l2]]
		bad  = [listOfmatrices[x] for x in I[l2:]]
		#print (str([len(x.shape) for x in bad]))
		#print (str([len(x.shape) for x in good]))
		return [good , bad]


	'''
	trimAll            -------------
	'''
	def trimAll(self):
		for siz in range(len(self.theList)):
			self.trim(siz)

	'''
	trim            -------------
	'''
	def trim(self,size):
		if (len(self.theList[size]) > 4.5*self.l) and (ran.random() < 0.95):
			[good, empt] = self.chooseLbest(self.theList[size],self.l);
			self.theList[size] = good;
			self.ocupations[size] = len(good);

	'''
	addMatrices            -------------
	'''
	def addMatrices(self,matrices):

		sizes  = np.array([B.k for B in matrices]);
		usizes = np.unique(sizes);


		for size in usizes:
			thisSize    = [B for B in matrices if B.k == size]
			minNormTS = min([B.norm() for B in thisSize]);
			if (minNormTS < self.smallestEver[size]):
				self.smallestEver[size]=minNormTS;

			allThisSize = self.theList[size]+thisSize;
			allThisSize,EMPT = self.chooseLbest(allThisSize,len(allThisSize));
			#allThisSize,EMPT = self.chooseLbest(allThisSize,2*self.l);
			self.theList[size] = allThisSize;
			self.bestNorms[size]  = min([b.norm0() for b in allThisSize]);
			self.ocupations[size] = len(self.theList[size]);


	'''
	smallestNonEmpty       -------------
	'''
	def smallestNonEmpty(self):
		ocupied = np.nonzero(self.ocupations)[0].tolist();
		ocupied = [ x for x in ocupied if x >= self.minSize ]
		ocupied.sort()
		if len(ocupied)==0:
			return self.maxSize - 1;
		return ocupied[0];

	def restartMulticore(self):
		self.multiCoreHelper.kill();
		gc.collect();
		self.multiCoreHelper = amp.auxForMultiCore(self.numCores)
