'''
Author:          Victor Mireles - Freie Universitaet Berlin
Email:           syats.vm@gmail.com

Latest version:  https://github.com/syats/ModuleReusability

Distributed under GPL V3.0
'''


import networkx as nx
import numpy as np
import random as ran
import auxSetFunctions as sf
#import graphCoverCalculatorILP as agc
#import graphCoverCalculatorILP as gc
import generateSimpleMatrix as gsm
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.sparse import lil_matrix


#-------------------------------------------------
#
#Creates the set of base modules, that includes the constitutve an tissue-specific genes
def constitutiveAndGeneSpecific(C,G):

	n = C.shape[0]; #Number of tissues
	m = C.shape[1]; #Number of genes
	allNodes = range(m);
	addedNodes = 0;
	Bbase=[];

	for j in range(n+1):
		Bbase.append([]);

	for x in range(m):
		#If the gene is tissue specific
		if (C[:,x].sum()==1):
			for tn in range(n):
				if (C[tn,x]==1):
					Bbase[tn+1].append(x);
					addedNodes+=1;
					allNodes.remove(x);

		if (C[:,x].sum()==n):
			Bbase[0].append(x);
			addedNodes+=1;
			allNodes.remove(x);

	B=[];
	for bt in Bbase:
		subG = G.subgraph(bt);
		CCs = nx.connected_components(subG);
		for cc in CCs:
			B.append(cc);

	print(" \tV-const&tspecif: "+str(addedNodes ));
	return [Bbase,allNodes,addedNodes];



#-----------------------------------------------------
#
# Puts a set of nodes into a set of k random molues
def generateRandomModules(k,allNodes):
	numNodes = len(allNodes);

	if (k>numNodes):
		k = numNodes
	moduleNumbers = range(k);  #These are the modules for the rest of the genes

	Bext=[];
	for j in range(k+3):
		Bext.append([]);

	allNodes2 = list(allNodes);
	ran.shuffle(allNodes2)
	modsize = int(numNodes / k)+1



	print str(k)+" "+str(numNodes)+" "+str(modsize)
	inNode = 0;
	j=0;
	while (inNode < numNodes):
		Bext[j] = allNodes2[inNode:(inNode + modsize)]
		inNode = inNode + modsize;
		j=j+1;

	return Bext

#-----------------------------------------------------
#
# Puts a set of nodes each into a singleton module
def generateSingletonModules(k,allNodes):

	moduleNumbers = range(len(allNodes));  #These are the modules for the rest of the genes
	Bext=[];
	for j in range(len(allNodes)):
		Bext.append([]);
	j=0;
	for x in allNodes:
		Bext[j].append(x);  # we assign it into a singleton module
		j+=1;



	return Bext;


#-----------------------------------------------------
#
# For every node, a module containing its neighbors and itself
def generateNeighborsAsModules(G,allNodes):
	Bext=[];
	for x in allNodes:
		xNeighbors = G.neighbors(x);
		xNeighbors.append(x);
		Bext.append(xNeighbors);
	return Bext;


#-----------------------------------------------------
#
#
def generateKmeansAsModules(G,K,C):

	C2 = C.copy();
	geneFeaturesArray = whiten(np.array(C2.T))
	numTissues = C.shape[0];
	n = C.shape[0]; #Number of tissues
	m = C.shape[1]; #Number of genes


	k = K
	#we cluster C using k means with this k
	[codeBook , disto] = kmeans(geneFeaturesArray,k,iter=50);
	#this returns a set of centroids
	#for every node, we find what centroid dominates it
	[clusterBelonging , disto] = vq(geneFeaturesArray,codeBook);
	#we now have k blocks
	Bt=[];
	for j in range(k):
		Bt.append([]);
	for j in range(m):
		Bt[clusterBelonging[j]].append(j);
	#for every block, we partition it into connected components, and put each into a block

	B=[];
	ratios = [];
	for bt in Bt:
		subG = G.subgraph(bt);

		CCs = nx.connected_components(subG);
		if len(bt)>0:
			#print(str(len(bt))+" "+str(len(CCs))+" "+str(len(subG.edges())));
			ratios.append(len(bt)/len(CCs));
		for cc in CCs:
			B.append(cc);

	return B


#-----------------------------------------------------
#
# For every edge of a graph, returns a module
def generateEdgesAsModules(G,allNodes):
	nonSingletons = list(allNodes);
	Bext=[];
	for ed in G.edges(): #The edges
		numNode1 = ed[0];
		numNode2 = ed[1];
		if ((numNode1 in allNodes) | (numNode2 in allNodes)):
			Bext.append([numNode1,numNode2]);
			if ((numNode1 in nonSingletons)):
				nonSingletons.remove(numNode1);
			if ((numNode2 in nonSingletons)):
				nonSingletons.remove(numNode2);
	Bisolated = generateSingletonModules(len(allNodes),allNodes)
	return Bext+Bisolated;



#This small function computes the difference between C and S_k * B_k
def testBk(BkO,SkO,CO):

	if (BkO.shape[0] <= BkO.shape[1]):

		C = CO
		Bk = BkO
		Sk = SkO
	else:
		C  = CO.T;
		Bk = BkO.T;
		Sk = SkO.T
	Cstar = np.dot(Sk , Bk);
	Cstar[np.nonzero(Cstar>1)] = 1;
	diff =  abs((C - Cstar)).sum();
	return diff








#-----------------------------------------------------
#
# GRAPHLESS "Add one more module from *one of the best* redundant modules"
#
# Takes a block matrix and, for every pair of blocks, computes their intersection.
# It also takes a parameter q.
# From the list of pairs of blocks, it takes the q'th
#   (q=0 being the pair that intersects the most)
# and i) removes this intersecting part from both and ii) makes a new block that contains only this intersection
#
# This effectively increases the number of rows in the matrix by one
#
# Returns this new matrix, as well as a number 		moreLeft that indicates if it is possible to get more

def oneMoreBlockRemovingQthOverlap(B,weightMatrix,q=0):
	m = B.shape[1]; #Number of genes
	originalK = B.shape[0];

	Bk = B.copy();


	blockSizes  = Bk.sum(axis=1);
	goodIndices = np.nonzero(blockSizes)[0];
	goodIndices.sort();
	Bk = Bk[goodIndices,:];

	iters = 0;

	#This cycle adds a row to the matrix, but it can also remove rows if it turns out that a row is a propper subset of another.
	while (Bk.shape[0] < originalK+1):
		iters+=1
		currentK = Bk.shape[0];
		[best1,best2,moreLeft] = bestIntersectingRows(Bk,q,weightMatrix)

		#This means there are no more block intersections
		if (best1 == best2):
			#print("NoMBi "+str(iters)+"N"+str(Bk.sum())+"    "),
			return [Bk , -1]

		#print(str(best1)+" \t "+str(best2))

		#Then we find the set q of indices in which they both have ones
		row1 = Bk[best1,:]
		row2 = Bk[best2,:]
		qq = np.nonzero((row1 + row2) == 2)[0];
		#print(str(len(qq)))
		#We create a new row that has ones only in q
		newRow = np.zeros([1,m]);
		newRow[0,qq] = 1;
		#print("newRowSum= "+str(newRow.sum()))

		#And we put 0s in the two original rows in q
		Bk[best1,qq] = 0;
		Bk[best2,qq] = 0;

		sumNewRow = newRow.sum();
		subBk = Bk[:,qq]
		sumSubBk = subBk.sum(axis=1)
		ind = np.nonzero(sumSubBk==sumNewRow)[0];
		Bk[np.ix_(ind,qq)] = 0

		Bk = np.concatenate((Bk,newRow));




		#print("norm of Bk after removing common and adding newrow= "+str(Bk.sum()))
		#we must make sure that the matrix contains no block that is a propper subset of another, if so, we remove the smaller one
	#For each of the new blocks, we check if it is properly contained somewhere. Since these blocks are subsets of former blocks, they can't properly contain any other (because we assume this didnt hapen in the original B)

		Bk = removeContainedFromB(Bk)



	#if (iters > 1):
	#	print("I: "+str(iters)+"N"+str(Bk.sum())+"   "),
	if (Bk.sum()>=B.sum()):
		print(str(B.shape)+" -> "+str(Bk.shape)+" INCREASE IN NORM!!! "+str(iters));
	return [lil_matrix(Bk,dtype='uint8') , moreLeft]

def bestIntersectingRows(Bk,q,weightMatrix):
	moreLeft = 0;
	#First we find the two rows with the largest dot product, but whose dot product is still small than the norm of any of them
	coincidenceMatrix = np.dot(weightMatrix*Bk , (Bk.T));
	#coincidenceMatrix = np.dot(Bk , (Bk.T));
	coincidenceMatrix = coincidenceMatrix - np.diag(np.diag(coincidenceMatrix));
	R = coincidenceMatrix.reshape(coincidenceMatrix.shape[0]*coincidenceMatrix.shape[1],1);
	#if coincidenceMatrix.sum()==0:
		#print("HERE IT ENDS: "+str(currentK)+" SUM= "+str(Bk.sum()))
	possibleOverlaps = np.unique(R);
	possibleOverlaps.sort();
	if q >= len(possibleOverlaps):
		q = len(possibleOverlaps) - 1;
		moreLeft = -1;

	q = q + 1; #Because of numpys backward's indexing.. [-a:] returns the last a elements
	#		print(str(len(possibleOverlaps))+" q "+str(q))
	qthOverlap = possibleOverlaps[-q:][0];
	if (qthOverlap == 0):
		qthOverlap = possibleOverlaps[-1:][0]

	#print(str(qthOverlap))
	entriesWithThisOverlap = np.nonzero(coincidenceMatrix==qthOverlap);
	numEntriesThisOverlap  = len(entriesWithThisOverlap[0]);
	chosenPairIndex = ran.choice(range(numEntriesThisOverlap));
	moreLeft = numEntriesThisOverlap-1;

	best1 = entriesWithThisOverlap[0][chosenPairIndex];
	best2 = entriesWithThisOverlap[1][chosenPairIndex];
	if (best1 == best2):
		print("*"+str(q)+"/"+str(len(possibleOverlaps))+" :"+str(qthOverlap));
	return [best1,best2,moreLeft]


#Rows of Bk are blocks, columns are genes
def removeContainedFromB(oBk):

	Bk = oBk

	someContained = True;

	while(someContained):
		coincidenceMatrix = np.dot(Bk , (Bk.T));
		k = coincidenceMatrix.shape[0]
		#coincideMatrix[j1,j2] = J1 dot J2 tells the number of genes shared between block j1 and block j2
		sizes = Bk.sum(axis=1);
		sizeMatrix = np.array([sizes,]*k)
		#sizeMatrix[j1,j2] = number of genes in block j2

		differences = sizeMatrix - coincidenceMatrix ;
		differences = differences + np.diag([10 for i in range(differences.shape[0])])
		#differences[j1,j2] number of genes in j2 not in j1, except if j1==j2, then 10

		contained = np.nonzero(differences == 0)
		#j1,j2 in contained => all genes in j2 are also in j1, ie j2 \subset j1

		numCont = len(contained[0]);

		someContained = (numCont > 0)
		if (not someContained):
			return Bk
		for c in range(numCont):
			small = contained[1][c];
			big   = contained[0][c];
			qq = np.nonzero(Bk[small,:])[0];
			#print(str(len(qq))+" ;; big[qq]: "+str(Bk[big,qq].sum())+" small: "+str(Bk[small,:].sum()))
			Bk[big,qq] = 0;
		sizes = Bk.sum(axis=1);
		goodIndices = np.nonzero(sizes)[0];
		Bk = Bk[goodIndices,:];

	return Bk

#Given a matrix of blocks, removes any blocks that are contained in other
def removeContainedFromB_old(Bk):

	#This variable will be 0 only when there is no block whicih is a subset of another
	#After removing some of this subsets, we must check again because the removing process might have created more subsets
	someContained = 1;

	while(someContained>0):
		#toCheck = [best1, best2, Bk.shape[0]-1];

		coincidenceMatrix = np.dot(Bk , (Bk.T));
		propCont = set();

		for j1 in range(coincidenceMatrix.shape[0]):
			#print(str(j1)+" ->> "+str(coincidenceMatrix[j1,j1])+" "+str(Bk[j1,:].sum()))
			if Bk[j1,:].sum() == 0:
				continue
			for j2 in range(coincidenceMatrix.shape[1]):
				if (j1 == j2):
					continue
				if (coincidenceMatrix[j1,j2] == coincidenceMatrix[j1,j1]):  #All the elements of j1 are also in j2
					propCont.add(tuple([j1,j2]));

		someContained = len(propCont)
		#print("Number of prop contained "+str(len(propCont))+" "+str(Bk.shape))


		#Every time a small block is contained in a big block, we make all the entries of the big one that are contained in the small one, zero.
		for tup in propCont:
			#print(str(tup)+" "),
			big   = tup[1];
			small = tup[0];
			qq = np.nonzero(Bk[small,:])[0];
			#print(str(len(qq))+" ;; big[qq]: "+str(Bk[big,qq].sum())+" small: "+str(Bk[small,:].sum()))
			Bk[big,qq] = 0;

		#print("norm of Bk after removing properly contained "+str(Bk.sum()))
		#Some block might now be empty, we have to check for that, and remove those which are
		blockSizes  = Bk.sum(axis=1);
		goodIndices = np.nonzero(blockSizes)[0];
		goodIndices.sort();
		Bk = Bk[goodIndices,:];
		return Bk;


#-----------------------------------------------------
#
# GRAPHLESS "Add one more module by merging the largest rectangle"
#
#
# This effectively increases the number of rows in the matrix by one
#
# Returns this new matrix, as well as a number 		moreLeft that indicates if it is possible to get more

def oneMoreBlockByRectangleMerging(BB,weightMatrix,Q=0,searchFirstRows=1):
	#We work with B being m times k  with k<=m
	inv = 0
	if (BB.shape[0] < BB.shape[1]):
		B = BB.T;
		inv = 1;
	else:
		B = BB;

	m = B.shape[0];
	k = B.shape[1];

	originalK = B.shape[1];
	Bk = B.copy();
	currentK = Bk.shape[1];

	if (searchFirstRows==0):
		Bk = Bk.T

	q = Q
	if (q > Bk.shape[1]):
		q = Bk.shape[1]-1

	while (currentK < originalK+1):

		foundGoodX = 0;
		tryNum = -1;
		while (foundGoodX == 0):
			tryNum += 1;
			[best1,best2,moreLeft] = bestIntersectingRows(Bk,q,weightMatrix)
			sums = Bk[[best1,best2],:].sum(axis=0)
			J0 = np.nonzero(sums==2)[0];


			#_-----------------_
			sumsPerCol = Bk.sum(axis=0)
			best = np.argsort(sumsPerCol).tolist();
			best.reverse();
			lim = len(best)-(q+tryNum)
			J0 = best[0:lim]


			#We find all unique sets of columns that, for some row, are all ones.
			Br = Bk[:,J0];
			uniqueRows = set()
			for x in range(Br.shape[0]):
				uniqueRows.add(tuple(np.nonzero(Br[x,:])[0]));


			#Of all sets of columns, we will try to find the one that is the base of the largest rectangle: i.e., that has the most rows that are all ones in that set. Lets say the set that maximixes this is of size sizeOfSet, and the number of rows that have all ones in this set of columns if numRows If we then remove these entries and make a new column that has ones only in those rows, then we would have removed as much as numRows x sizeOfSet - numRows
			maxL = -1;
			maxC = -1;
			maxX = set();
			for l in uniqueRows:
				lenL = len(l)
				if (lenL < 2):
					continue
				X = set(); #The set of rows that have all ones in l
				for x in range(Br.shape[0]):
					if Br[x,l].sum() == lenL:
						X.add(x);
				countl = len(X)*(lenL-1)
				if (countl > maxC) and (len(X) > 1):
					maxC = countl;
					maxL = l;
					maxX = list(X);
			if len(maxX)>1:
				foundGoodX = 1

		#It might be that the rectangle has a basis actually extending beyond J0 (because our choose of J0 is not necesarily optimal)
		#We will now find the set of columns that have all-ones in the rows stored in maxX.
		sums = Bk[maxX,:].sum(axis=0)
		L = np.nonzero(sums==len(maxX))[0]
		#print(str(len(L))+" "+str(len(maxX))+"  "+str(Bk[np.ix_(maxX,L)].sum()))

		if (searchFirstRows==1):
			newCol = np.zeros([m,1]);
			newCol[maxX,0] = 1;
			Bk[np.ix_(maxX,L)] = 0;
			Bk = np.concatenate((Bk,newCol),1);
			Bk = removeContainedFromB(Bk.T).T
			currentK = Bk.shape[1];

		else:
			newCol = np.zeros([1,m]);
			newCol[0,L] = 1;
			Bk[np.ix_(maxX,L)] = 0;
			Bk = np.concatenate((Bk,newCol));
			Bk = removeContainedFromB(Bk)
			currentK = Bk.shape[0];

	if (inv == 1):
		Bk = Bk.T

	if (searchFirstRows==0):
		Bk = Bk.T

	return Bk
	#return [Bk , 1]


#-----------------------------------------------------
#
# GRAPHLESS "Add one more module from redundant modules"
#
# Takes a block matrix B with k rows (blocks) and finds the two blocks with highest overlap (number of genes in common)
# It returns a new matrix that has a new block with exactly that overlap, and removes the genes in commmon from the first two blocks, it also returns the indices of these two blocks
# This effectively creates a matrix with smaller zero norm
# If P = 1, then we guarantee only the best coincidence will be removed. If P < 1, there is certain chance that we don't remove the best coincidence

def oneMoreBlockRemovingOverlap(C,B,P=1):
	n = C.shape[0]; #Number of tissues
	m = C.shape[1]; #Number of genes
	Bk = B.copy();

	originalK = Bk.shape[0];

	#This cycle adds a row to the matrix, but it can also remove rows if it turns out that a row is a propper subset of another.
	while (Bk.shape[0] < originalK+1):
		currentK = Bk.shape[0];
		isSubset = np.zeros(currentK)
		#First we find the two rows with the largest dot product, but whose dot product is still small than the norm of any of them
		coincidenceMatrix = np.dot(Bk , (Bk.T));
		best1 = -1;
		best2 = -1;
		bestCoincidence = 0;
		p = 0;

		for j1 in range(currentK):
			for j2 in range(j1+1,currentK):
				if ((coincidenceMatrix[j1,j2] > bestCoincidence) and (coincidenceMatrix[j1,j2] < coincidenceMatrix[j1,j1])  and (coincidenceMatrix[j1,j2] < coincidenceMatrix[j2,j2])):
					if (p<P):
						bestCoincidence = coincidenceMatrix[j1,j2];
						best1 = j1;
						best2 = j2;
						p = ran.random()
				if (coincidenceMatrix[j1,j2] == coincidenceMatrix[j1,j1]):
					isSubset[j1] = isSubset[j1]+1;

		if ((best1 < 0) or (best2 < 0)):
			return 0
		#Then we find the set q of indices in which they both have ones
		row1 = Bk[best1,:]
		row2 = Bk[best2,:]
		q = np.nonzero((row1 + row2) == 2)[0];
		#We create a new row that has ones only in q
		newRow = np.zeros([1,m]);
		newRow[0,q] = 1;

		#And we put 0s in the two original rows in q
		Bk[best1,q] = 0;
		Bk[best2,q] = 0;
		Bk = np.concatenate((Bk,newRow));

		#we must make sure that the matrix contains no block that is a propper subset of another, if so, we remove the smaller one
		propCont = np.nonzero(isSubset)[0];
		if len(propCont) > 0:
			indices = set(range(Bk.shape[0]))
			goodIndices = indices - set(propCont);
			Bk = Bk[goodIndices,:];

	return [Bk , best1 , best2]


#-----------------------------------------------------
#
# GRAPHLESS. "The procedure"
# Uses the simple heuristic to come up with increasingly large amounts of modules
# For K = n it returns C
# For K > n, it takes the two blocks that, for K-1 share the most nodes, removes this common nodes from the blocks and creates a whole new node containing them
def theProcedure(C,K):
	n = C.shape[0]; #Number of tissues
	if (n==K):
		return C;
	m = C.shape[1]; #Number of genes

	#For k = n, it is trivial
	Bk = C.copy();
	Sk = np.identity(n);
	print(str(n)+" \t "+str(testBk(Bk,Sk,C))+" \t "+str(Bk.sum())+" \t 0");

	#If P = 1, then we guarantee only the best coincidence will be removed. If P < 1, there is certain chance that we don't remove the best coincidence
	P = 1;

	for k in range(n+1,K+1):

		#First we compute how many blocks are subset of another
		currentK = Bk.shape[0];
		isSubset = np.zeros(currentK)
		coincidenceMatrix = np.dot(Bk , (Bk.T));
		for j1 in range(currentK):
			for j2 in range(j1+1,currentK):
				if (coincidenceMatrix[j1,j2] == coincidenceMatrix[j1,j1]):
					isSubset[j1] = isSubset[j1]+1;

		#Then we compute the matrix with one added block
		[Bk , best1 , best2] = oneMoreBlockRemovingOverlap(C,Bk,P)

		#Now, we sum the two corresponding cols of S
		col1 = Sk[:,best1];
		col2 = Sk[:,best2];
		coincide = np.nonzero((col1+col2)>0)[0];
		newCol = np.zeros([n,1]);
		newCol[coincide,0] = 1;

		#And that is the new column for Sk+1
		Sk = np.concatenate((Sk,newCol),axis=1);


		print (str(k)+" \t "+str(testBk(Bk,Sk,C))+" \t "+str(Bk.sum())+" \t "+str(np.nonzero(isSubset)[0].shape[0]));

		#We check now the ILP-obtained S
		#ILPS = agc.coverWholeCWithBMatrix(C,Bk);
		#diff =  abs((Sk - ILPS)).sum();
		#fitILP = testBk(Bk,ILPS,C);
		#print (str(diff)+" "+str(fitILP));



	return Bk;
