'''
Author:          Victor Mireles - Freie Universitaet Berlin
Email:           syats.vm@gmail.com

Latest version:  https://github.com/syats/ModuleReusability

Distributed under GPL V3.0
'''



import math
import numpy as np
import random as ran
import scipy.misc as mps




def fromRtoC(R,m):
	C = np.zeros([m,R.shape[1]])
	for x in range(m):
		vn = ran.choice(range(R.shape[0]))
		C[x,:] = R[vn,:];
	#print(" !!!!!!!!!! "+str(R.shape))
	return C

# C is a matrix with m rows and n columns
#r is the number of different rows of C
#r must be less than 2^n - 1  (because that's the number of possible non zero vectors of size n)
#r must be smaller or equal than m
def generateRandomCMatrix(n,m,R_obj,minDensity=0,maxDensity=1):
	r_obj = R_obj
	if (r_obj > m):
		r_obj = m

	r = 0;
	max_tries = int(1+R_obj/2)*(150+int(maxDensity*100));
	criticalTries = int(max_tries / 2)+1;
	randomizingTries = max([int(criticalTries/20)+1,20]);

	print("initial "+str(max_tries))

	minNumOnes = max([n,r_obj])
	maxNumOnes = n*r_obj
	rangeNumOnes = maxNumOnes-minNumOnes;
	print(str(int(minNumOnes+minDensity*rangeNumOnes))+" : "+str(int(minNumOnes+maxDensity*rangeNumOnes)));
	density = ran.choice(range(int(minNumOnes+minDensity*rangeNumOnes),int(minNumOnes+maxDensity*rangeNumOnes)))
	print(density)

	baseVectors = np.zeros([r_obj,n]);
	for i in range(int(density*1.10)):
		x = ran.choice(range(r_obj));
		y = ran.choice(range(n));
		baseVectors[x,y] = 1
	#5 checks:  nonzero rows, nonzero columns, r different rows, full column rank, density
	Checks = 0;
	while Checks < 5:

		nonzeroRows = 0;
		nonzeroColumns = 0;
		differentRows = 0
		columnRank = 0
		densityC = 0 ;

		if ( max_tries % randomizingTries == 1):
			print("random "+str(max_tries))
			baseVectors = np.zeros([r_obj,n]);
			for i in range(density+10):
				x = ran.choice(range(r_obj));
				y = ran.choice(range(n));
				baseVectors[x,y] = 1


		if (max_tries < criticalTries):
			#print("CCCC");
			density = max([int(0.95*density), minNumOnes+1]);

	#while (r < r_obj) or (N < n):
		#print(str("*"))
		max_tries -= 1;

		if (baseVectors.shape[0] < r_obj):
			missingVectors = np.zeros([r_obj - baseVectors.shape[0],baseVectors.shape[1]]);
			baseVectors = np.concatenate((baseVectors,missingVectors))

		if (baseVectors.shape[1] < n):
			missingVectors = np.zeros([baseVectors.shape[0],n - baseVectors.shape[1]]);
			baseVectors = np.hstack((baseVectors,missingVectors))

		#First we put some ones to make sure that all tissues have at least one gene
		emptyTissues = 	np.nonzero(baseVectors.sum(axis=0)==0)[0]
		#print(str(emptyGenes))
		#print(str(baseVectors.shape))
		for i in emptyTissues:
			vn = ran.choice(range(r_obj))
			baseVectors[vn,i] = 1;

		emptyRs	= np.nonzero(baseVectors.sum(axis=1)==0)[0]
		#print(str(emptyRs))
		#print(str(baseVectors.shape))
		for vn in emptyRs:
			i = ran.choice(range(n))
			baseVectors[vn,i] = 1

		#print(str(baseVectors))
		#Now we make sure it has the correct amount of ones
		emptySpots = np.nonzero(baseVectors==0);
		#print(str(len(emptySpots))+" "+str(len(emptySpots[0])))
		indices = range(len(emptySpots[0]));
		ran.shuffle(indices);
		I = 0;
		origDensity = baseVectors.sum()
		#print(str(origDensity)+"\t\t"+str(density))
		target = density
		if (max_tries < criticalTries):
			target = 2*density
		while (origDensity + I < target) and (I < len(emptySpots[0])):
			baseVectors[emptySpots[0][I],emptySpots[1][I]] = 1;
			I += 1;

		#It might have been that two base vectors are the same, we must removethem
		similarity = np.dot(baseVectors,baseVectors.T)
		allIndices = np.ones(r_obj);
		#print(str(baseVectors))
		#print(str(similarity))
		for b1 in range(r_obj):
			for b2 in range(b1+1,r_obj):
				if (similarity[b1,b2] == similarity [b2,b2]) and (similarity[b1,b1] == similarity[b1,b2]):
					allIndices[b1] = 0
		baseVectors = baseVectors[np.nonzero(allIndices)[0],:];

		if (max_tries > 0):
			#It might also be that one tissue is a subset of the other, we remove an element from the large one
			someFound = 1
			while (someFound == 1):
				someFound = 0;
				for t1 in range(baseVectors.shape[1]):
					for t2 in range(baseVectors.shape[1]):
						if (t1==t2):
							continue
						rowsInCommon = [x for x in range(baseVectors.shape[0]) if (baseVectors[x,t1]==1 and baseVectors[x,t2]==1)]
						if (len(rowsInCommon) > 0):
							ss = 1
							if (max_tries < criticalTries):
								ss = max([int(len(rowsInCommon))-1 , 1])
							if (len(rowsInCommon) == baseVectors[:,t1].sum()):
								baseVectors[ran.sample(rowsInCommon,ss) ,t2] = 0
								someFound = 1
							if (len(rowsInCommon) == baseVectors[:,t2].sum()):
								baseVectors[ran.sample(rowsInCommon,ss) ,t1] = 0
								someFound = 1


		r = baseVectors.shape[0];
		N = baseVectors.shape[1];
		#print(str(baseVectors.shape))

		''' ------------ '''
		if (someFound == 0):
			columnRank = 1;

		''' ------------ '''
		if (r == r_obj):
			differentRows = 1

		''' ------------ '''
		emptyRows = 	np.nonzero(baseVectors.sum(axis=0)==0)[0]
		if (len(emptyRows)==0):
			nonzeroRows = 1;

		''' ------------ '''
		emptyColumns = 	np.nonzero(baseVectors.sum(axis=1)==0)[0]
		if (len(emptyColumns)==0):
			nonzeroColumns = 1;

		''' ------------ '''
		if (baseVectors.sum() >= density):
			densityC = 1;

		Checks = nonzeroRows + nonzeroColumns + differentRows + columnRank + densityC;

		#print(str(baseVectors))
		if(max_tries == 0):
			print("MMMM  "+str(baseVectors.sum()))

			Checks = 100;

	print(str(nonzeroRows) + str(nonzeroColumns) + str(differentRows) + str(columnRank)+ str(densityC))
	print("left "+str(max_tries))
	return fromRtoC(baseVectors,m)





#Generates a block matrix B of size (m x k) with the given overlap between blocks. It then generates an expression matrix C = BS (of size m x n) for some (k x n) matrix S, such that the number of different rows of of C is r_obj
def generateSimpleMatrix(n,m,k,r_obj,overlap):


	#First we generate disconnected blocks
	intersect = 1;
	while (intersect > 0):
		B = np.zeros([m,k]);
		cuttingPoints = range(1,m-1);
		ran.shuffle(cuttingPoints);
		cuttingPoints = cuttingPoints[0:k-1];
		cuttingPoints.sort();

		start = 0;
		for j in range(k):
			if (j < k - 1):
				end = cuttingPoints[j];
			else:
				end = m + 1;

			B[start:end,j] = 1;
			start = end;

		#Then we fill up overlap zeros at random
		#We get the zeros
		zerosR = np.nonzero(B==0)[0];
		zerosC = np.nonzero(B==0)[1];
		numZeros = len(zerosR);
		zerosI = range(numZeros);
		#and shuffle them
		ran.shuffle(zerosI);
		zerosR = zerosR[zerosI];
		zerosC = zerosC[zerosI];

		p = 0;
		while (B.sum() < m + overlap):
			B[zerosR[p],zerosC[p]] = 1;
			p += 1;

		#This is to guarantee that the B matrix is full rank (no block is a subset of another)
		intersect = 0;
		for j1 in range(k):
			for j2 in range(j1+1,k):
				inCommon = np.dot(B[:,j1],B[:,j2]);
				if (inCommon == B[:,j1].sum()) or (inCommon == B[:,j2].sum()):
					intersect += 1

	print("Created B");
	tryNo = 0;
	#Now we make up an S matrix
	intersect = 1;
	while (intersect > 0):

		S = np.zeros([k,n]);
		for i in range(n):
			j = ran.randint(0,k-1)
			S[j,i] = 1;
		for j in range(k):
			if S[j,:].sum() == 0:
				i = ran.randint(0,n-1)
				S[j,i] = 1;

		r = -1;
		while (r < r_obj):
			print(str(tryNo))
			tryNo += 1;
			if tryNo >= 50:
				return generateSimpleMatrix(n,m,k,r_obj,overlap)

			zerosR = np.nonzero(S==0)[0];
			zerosC = np.nonzero(S==0)[1];
			numZeros = len(zerosR);
			zerosI = range(numZeros);
			#and shuffle them
			ran.shuffle(zerosI);
			zerosR = zerosR[zerosI];
			zerosC = zerosC[zerosI];

			S[zerosR[0],zerosC[0]] = 1;
			C = np.dot(B,S);
			C[np.nonzero(C>0)] = 1

			if (C.sum() > (9.0/10.0)*m*n):
				S = np.zeros([k,n]);
				for i in range(n):
					j = ran.randint(0,k-1)
					S[j,i] = 1;
				for j in range(k):
					if S[j,:].sum() == 0:
						i = ran.randint(0,n-1)
						S[j,i] = 1;

			allGenes = set();
			for x in range(m):
				allGenes.add(tuple(C[x,:]))
			r = len(allGenes)

		#This is to guarantee that the C matrix is full rank (no tissue is a subset of another)
		intersect = 0;
		for i1 in range(n):
			for i2 in range(i1+1,n):
				inCommon = np.dot(C[:,i1],C[:,i2]);
				if (inCommon == C[:,i1].sum()) or (inCommon == C[:,i2].sum()):
					intersect += 1

	return [B,C,S]




def checkRowSum(C):
	sums = C.sum(axis=1)
	if len(np.nonzero(sums==0)[0])>0:
		return False
	return True

def checkColSum(C):
	sums = C.sum(axis=0)
	if len(np.nonzero(sums==0)[0])>0:
		return False
	return True

def checkColumnRank(C):
	sim = np.dot(C.T,C);
	siz = C.sum(axis=0);

	numCols = C.shape[1];
	for c1 in range(numCols):
		for c2 in range(c1+1,numCols):
			if (sim[c1,c2]==siz[c1]) or (sim[c1,c2]==siz[c2]):
				return False
	return True


def checkRowEquality(C):
	sim = np.dot(C,C.T);
	siz = C.sum(axis=1);

	numRows = C.shape[0];
	for r1 in range(numRows):
		for r2 in range(r1+1,numRows):
			if (sim[r1,r2]==siz[r1]) and (sim[r1,r2]==siz[r2]):
				return False
	return True


def generateRIterations(n,r,d,numIters):
	checks = 0;
	iterNum = 0;
	bestDiff = n*r;
	while (iterNum < numIters) and (checks < 4):
		iterNum+=1
		checks = 0;
		R = generateR(n,r,d);
		if (checkColSum(R)):
			checks+=1;
		if (checkRowSum(R)):
			checks+=1;
		if checkRowEquality(R):
			checks+=1;

		diff = abs(d-R.sum())
		if (checks == 3):
			if diff < bestDiff:
				bestDiff = diff
				bestR = R

		if (diff==0):
			checks+=1;

	#print(str(checks)+" "+str(iterNum))
	return bestR


def binvect2dec(vect):
	n = vect.shape[0];
	nz = np.nonzero(vect)[0].tolist();
	powers = [math.pow(2,n-1-i) for i in nz]
	return(sum(powers));

def dec2binvect(num,n):
	vect = np.zeros(n);
	cur = num;
	for s in range(n):
		p = n-1-s;
		if cur >= math.pow(2,p):
			vect[s] = 1;
			cur = cur-math.pow(2,p)
	return vect


def initWithTooFrequentSizes(R1,C,sizes1,protectedZeros,protectedOnes):
	sizes=sizes1.copy()
	sizesf = sizes1.copy();
	R = R1.copy()
	r = R.shape[0];
	n = R.shape[1];

	usedBinNums = set();
	allrows     = range(r);
	ran.shuffle(allrows);


	maxS = int(max(sizes));
	toPlace = -1;
	for s in range(maxS+1):
		nts = len(np.nonzero(sizes==s)[0]);
		maxts = mps.comb(n,s);
		#print("i: "+str(s)+" num"+str(nts)+" max"+str(maxts)+" siz"+str(len(sizes)))

		if nts <= 0.9*maxts:
			continue;

		#print("*")
		#If there are too many rows with this size, we add them directly (because it otherwise it would take a very long time to find them by the random process)
		rowsThisS = [C[x,:] for x in np.nonzero(sizes==s)[0] ];
		toPlace = nts;
		for vect in rowsThisS:

			num = binvect2dec(vect);

			#if num in usedBinNums:
				#print("!!!!\t"),
				#continue
			for ro in allrows:
				if (protectedZeros[ro,:].sum()>0) and (vect[np.nonzero(protectedZeros[ro,:])].sum()>0):
					continue
				if (protectedOnes[ro,:].sum()>0) and (vect[np.nonzero(protectedOnes[ro,:])].sum() < protectedOnes[ro,:].sum()):
					continue

				R[ro,:] = vect;
				allrows.remove(ro);
				usedBinNums.add(num);
				toPlace -= 1;
				break

		if (toPlace == 0):
			sizesf = np.array([so for so in sizesf if so != s]);

	print("Rsum="+	str(R.sum())+ "sizes"+str(len(sizesf))+" rows:"+str(len(allrows))+" numsleft"+str(r-len(usedBinNums))+" toplace="+str(toPlace))
	return R,sizesf,usedBinNums,allrows


#Given C, generate a matrix with the same row-sum sequence. That is, if C has m1 rows whose sum is s1, then the returned matrix also has m1 rows whose sum is s1. Only the row-sum is preserved, so, under most conditions, the rows themselves will be different. Note that if C contains many rows of a certain, not too large sum, s1, then it can happen that it is fairly hard to find different rows of sum s1, so that these rows in the returned matrix are likely to be the same as in C
def generateRWithBinary(C):
	d = C.sum();
	r = max(C.shape);
	n = min(C.shape);
	if r > 2**n - 1:
		r = 2**n - 1
	if (d < n*(n-1)/2):
		d = n*(n-1)/2

	sizes = C.sum(axis=np.argmin(C.shape));


	#print(str(n)+" "+str(r)+" "+str(d))
	protectedZeros = np.zeros([r,n]);
	protectedOnes  = np.zeros([r,n]);
	R = np.zeros([r,n]);

	#The list of pairs of tissues
	listOfPairs = [];
	for t1 in range(n):
		for t2 in range(t1+1,n):
			numRsToDis = 1;
			listOfPairs.append(tuple([t1,t2,numRsToDis]));
	ran.shuffle(listOfPairs)

	#print("n:"+str(n)+" r:"+str(r)+" d:"+str(d)+" pairs:"+str(len(listOfPairs)))


	usedBinNums = set();
	#print("usedNum: "+ str(len(usedBinNums))+" nzr= "+str( len(np.nonzero((R.sum(axis=1)>0))[0]) ));

	R,sizes,usedBinNums,allrows = initWithTooFrequentSizes(R,C,sizes,protectedZeros,protectedOnes);
	ran.shuffle(sizes);

	unplaced = r-len(usedBinNums);
	print("\n");
	print("usedNum: "+ str(len(usedBinNums))+" nzr= "+str( len(np.nonzero((R.sum(axis=1)>0))[0]) ));
	#print("Rsum="+	str(R.sum())+ "sizes"+str(len(sizes))+" rows:"+str(len(allrows)))
	#print(len(usedBinNums))
	while(len(sizes) >= 1):
		#print(len(usedBinNums))
		#num = ran.randint(1,math.pow(2,n));
		#vect = dec2binvect(num,n);

		toPlace =1;
		triesLeft = 100;
		while((toPlace>0) and (triesLeft>0)):
			triesLeft -= 1;
			siz = int(sizes[0]);
			poss = ran.sample(range(n),siz);
			vect = np.zeros(n);
			vect[poss] = 1;
			num = binvect2dec(vect);
			#print(str(len(usedBinNums)));

			#if num in usedBinNums:
				#ran.shuffle(sizes);
				#print(str(siz)+" ub: "+str(len(usedBinNums)))
			#	continue

			for ro in allrows:
				if (protectedZeros[ro,:].sum()>0)and(vect[np.nonzero(protectedZeros[ro,:])].sum()>0):
					continue
				if (protectedOnes[ro,:].sum()>0) and (vect[np.nonzero(protectedOnes[ro,:])].sum() < protectedOnes[ro,:].sum()):
					continue

				R[ro,:] = vect;
				allrows.remove(ro);
				usedBinNums.add(num);
				sizes = sizes[1:]
				toPlace = 0;
				unplaced -= 1;
				break

	#print("toplace: "+str(unplaced))
	#print("ones  " +str(protectedOnes.sum())+" -> "+str(R[np.nonzero(protectedOnes)].sum()) )
	#print("zeros "+str(protectedZeros.sum())+" -> "+str(R[np.nonzero(protectedZeros)].sum()) )
	#print("Rsum="+	str(R.sum())+ "sizes"+str(len(sizes))+" rows:"+str(len(allrows)))
	#print("usedNum: "+ str(len(usedBinNums))+" nzr= "+str( len(np.nonzero((R.sum(axis=1)>0))[0]) ));
	#print(len(usedBinNums))

	#print("colsum");
	#print("z0 "+str(protectedZeros.sum())),
	#We make sure	every column has at least one 1
	if (not checkColSum(R)):
		#print("cols");
		sums = R.sum(axis=0)
		missingCols = np.nonzero(sums==0)[0]
		for c in missingCols:
			foundX = 0
			while(foundX ==0):
				if (R+protectedZeros).sum() == R.shape[0]*R.shape[1]:
					return R
				X = ran.choice(range(r))
				if protectedZeros[X,c]==0:
					foundX = 1;
			R[X,c] = 1;
			protectedOnes[X,c] = 1

	#print("rowsum");
	#We make sure	every row has at least one 1
	if (not checkRowSum(R)):
		#print("rows "),
		sums = R.sum(axis=1)
		missingRows = np.nonzero(sums==0)[0]
		print(str(len(missingRows)));
		for x in missingRows:
			foundC = 0
			while(foundC ==0):
				if (R+protectedZeros).sum() == R.shape[0]*R.shape[1]:
					return R
				C = ran.choice(range(n))
				if protectedZeros[x,C]==0:
					foundC = 1;
			R[x,C] = 1;
			protectedOnes[x,C]= 1

	return R


# !!! ------ FASTEST ------ !!!
#generates R, an r times n binary matrix with ~d 1's, such that:  1) no rows sum 0, 2) no cols sum zero 3) the nonzero entries of one colum are a subset of the nonzero entries of another. This last condition implies that for any two columns (i1,i2), there is a pair of rows x,y such that R[x,i1]=R[y,i2]=0  R[x,i2]=R[y,i1]=1
def generateR(n,r,d):
	if r > 2**n - 1:
		r = 2**n - 1
	if (d < n*(n-1)/2):
		d = n*(n-1)/2

	print("numberOfOnesInRandomMatrix: "+str(d))
	protectedZeros = np.zeros([r,n]);
	protectedOnes  = np.zeros([r,n]);
	R = np.zeros([r,n]);


	#The list of pairs of tissues
	listOfPairs = [];
	for t1 in range(n):
		for t2 in range(t1+1,n):
			numRsToDis = 1;
			listOfPairs.append(tuple([t1,t2,numRsToDis]));
	ran.shuffle(listOfPairs)

	#print("n:"+str(n)+" r:"+str(r)+" d:"+str(d)+" pairs:"+str(len(listOfPairs)))

	#For every pair of columns, we differtiate them
	for pair in listOfPairs:
		t1 = pair[0];
		t2 = pair[1];
		toMake = pair[2]

		for m in range(toMake):
			foundX = 0;
			while(foundX == 0):
				X = ran.choice(range(r));
				if (protectedZeros[X,t1] == 0) and (protectedOnes[X,t2] == 0) :
					foundX = 1;

			foundY = 0;
			while(foundY == 0):
				Y = ran.choice(range(r));
				if (X==Y):
					continue
				if (protectedZeros[Y,t2] == 0) and (protectedOnes[Y,t1] ==0) :
					foundY = 1;

			newOnes = 0;
			if R[X,t1] == 0:
				newOnes += 1;
			R[X,t1] = 1
			R[X,t2] = 0
			protectedZeros[X,t2] = 1
			protectedOnes[X,t1] = 1

			if R[Y,t2] == 0:
				newOnes += 1;
			R[Y,t2] = 1
			R[Y,t1] = 0
			protectedZeros[Y,t1] = 1
			protectedOnes[Y,t2] = 1

	#print("colsum");
	#print("z0 "+str(protectedZeros.sum())),
	#We make sure	every column has at least one 1
	if (not checkColSum(R)):
		sums = R.sum(axis=0)
		missingCols = np.nonzero(sums==0)[0]
		for c in missingCols:
			foundX = 0
			while(foundX ==0):
				if (R+protectedZeros).sum() == R.shape[0]*R.shape[1]:
					return R
				X = ran.choice(range(r))
				if protectedZeros[X,c]==0:
					foundX = 1;
			R[X,c] = 1;
			protectedOnes[X,c] = 1

	#print("rowsum");
	#We make sure	every row has at least one 1
	if (not checkRowSum(R)):
		sums = R.sum(axis=1)
		missingRows = np.nonzero(sums==0)[0]
		for x in missingRows:
			foundC = 0
			while(foundC ==0):
				if (R+protectedZeros).sum() == R.shape[0]*R.shape[1]:
					return R
				c = ran.choice(range(n))
				if protectedZeros[x,c]==0:
					foundC = 1;
			R[x,c] = 1;
			protectedOnes[X,c] = 1

	#print(" zm"+str(protectedZeros.sum())),
	#print("disctinctions")
	sim = np.dot(R,R.T);
	siz = R.sum(axis=1);

	impossible = 0;

	allProtected = protectedZeros + protectedOnes;
	for r1 in range(r):
		if (impossible == 1):
			print("impossible"+str(impossible))
			break
		for r2 in range(r1+1,r):
			if (sim[r1,r2]==siz[r1]) and (sim[r1,r2]==siz[r2]):
				#rows r1 and r2 are the same, we find among them a single entry that is not protected, and we toggle it.
				toggeable = [tuple([r1,x]) for x in np.nonzero(allProtected[r1,:]==0)[0] ]
				toggeable += [tuple([r2,x]) for x in np.nonzero(allProtected[r2,:]==0)[0] ]
				if (len(toggeable)) == 0:
					impossible=1;
					break
				ran.shuffle(toggeable)
				v = R[toggeable[0][0],toggeable[0][1]]
				if (v==0):
					R[toggeable[0][0],toggeable[0][1]] = 1;
					protectedOnes[toggeable[0][0],toggeable[0][1]] = 1
				else:
					R[toggeable[0][0],toggeable[0][1]] = 0;
					protectedZeros[toggeable[0][0],toggeable[0][1]] = 1
				allProtected[toggeable[0][0],toggeable[0][1]] = 1

	#print("impossible "+str(impossible))

	#print("\nn:"+str(n)+" r:"+str(r)+" d:"+str(d)+" pairs:"+str(len(listOfPairs)))
	#print(" zf "+str(protectedZeros.sum()))
	#print("fillup");
	totOnes = R.sum();
	stillZeros = np.nonzero(R==0);
	stillZerosX = stillZeros[0];
	stillZerosR = stillZeros[1];
	indices = range(len(stillZerosX))
	ran.shuffle(indices)
	zp = 0;
	while(totOnes < d):
		foundCords = 0
		while (foundCords==0):
			if (zp >= len(stillZerosX)) or ((R+protectedZeros+protectedOnes).sum() == R.shape[0]*R.shape[1]):
				print("\n ran out of space for ones\n");
				return R
			x = stillZerosX[indices[zp]];
			c = stillZerosR[indices[zp]];
			zp += 1;
			if (protectedZeros[x,c]==0) and (protectedOnes[x,c]==0):
				foundCords = 1
		R[x,c] = 1
		totOnes += 1
		#print("*"+str(totOnes)),

	print("Everything shoud be finished now");
	return R

def numberOfDifferentRows(R):
	allGenes = set();
	for x in range(R.shape[0]):
		allGenes.add(tuple(R[x,:]));
	return len(allGenes)

def generateRrepetitions(n,m,do):
	found = 0;
	d = do;
	if (do>1):
		d = do/float(m*n);

	expnnz = d*m*n;
	#print("expnnz "+str(expnnz));
	while(found==0):

		C = np.random.rand(m,n)
		nz = np.nonzero(C<d)
		C = np.zeros([m,n])
		C[nz[0],nz[1]] = 1;

		z=np.nonzero(C.sum(axis=1)==0)[0]
		for g in z:
			t = ran.randint(0,n-1);
			C[g,t] = 1

		z=np.nonzero(C.sum(axis=0)==0)[0]
		for t in z:
			g = ran.randint(0,m-1);
			C[g,t] = 1

		#Maybe we have too many ones
		toRem = int(C.sum() - expnnz);
		#if we are missing ones
		if (toRem < 0):
			nz = np.nonzero(C==0);
			ind = range(len(nz[0]));
			ran.shuffle(ind);
			for i in range(-toRem):
				if (i >= len(ind)):
						break;
				g = nz[0][ind[i]];
				t = nz[1][ind[i]];
				C[g,t] = 1;

		#If we have too many ones
		if (toRem>0):
			nz = np.nonzero(C==1);
			ind = range(len(nz[0]));
			ran.shuffle(ind);
			i = -1;
			while(toRem > 0):
				i += 1;
				if (i >= len(ind)):
					break;
				g = nz[0][ind[i]];
				t = nz[1][ind[i]];
				if ( C[g,:].sum() > 1 ) and ( C[:,t].sum() > 1 ):
					C[g,t] = 0;
					toRem -= 1;


		z1=len(np.nonzero(C.sum(axis=1)==0)[0])==0
		z0=len(np.nonzero(C.sum(axis=0)==0)[0])==0
		co=True;

		D = np.dot(C.T,C);

		for t in range(n):
			st = D[t,t].sum();
			for t2 in range(t+1,n):
				if (D[t2,t] == st):
					co=False;

		if (co and z1 and z0):
			#print("actnnz "+str(C.sum())+"\n\n");
			return C;

def generateRSS(Co,maxTries = 100):
	C = Co;
	if Co.shape[1] > Co.shape[0]:
		C = C.T
	m = C.shape[0];
	n = C.shape[1];
	done = 1
	tries = -1;
	R = np.zeros([m,n]);
	sizes = C.sum(axis=1);
	while (done > 0):
		tries += 1;
		R = np.zeros([m,n]);

		for x in range(m):
			siz = int(sizes[x]);
			if siz == 0:
				print("0 row!");
				continue
			cols = ran.sample(range(n),siz);
			R[x,cols]=1;

		if checkColumnRank(R.T) or (tries > maxTries):
			done = 0;
	newSiz = R.sum(axis=1);
	#print(str(np.abs(newSiz-sizes).sum()))
	return R;

def generateWithUniformRows(Co):
	C = Co;
	if Co.shape[1] > Co.shape[0]:
		C = C.T
	m = C.shape[0];
	n = C.shape[1];
	tot = C.sum();

	R = np.zeros(C.shape);
	perRow = int(tot/float(m));

	for x in range(m):
		cols = ran.sample(range(n),perRow);
		R[x,cols] = 1;

	missing = C.sum() - R.sum();
	if missing>0:
		stillZero  = np.nonzero(R==0);
		numNonzero = len(stillZero[0]);
		toSwitch   = ran.sample(range(numNonzero),int(missing));
		toSwitchX  = stillZero[0][toSwitch];
		toSwitchY  = stillZero[1][toSwitch];
		R[toSwitchX,toSwitchY] = 1;

	return R
