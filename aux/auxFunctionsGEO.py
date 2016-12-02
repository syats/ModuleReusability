'''
Author:          Victor Mireles - Freie Universitaet Berlin
Email:           syats.vm@gmail.com

Latest version:  https://github.com/syats/ModuleReusability

Distributed under GPL V3.0
'''


import numpy as np

import csv

def load_Labeled_csv(fileName,thr=1,sep='\t',readStart=1):

	proteinsFile = fileName
	expre = dict();
	with open(proteinsFile, 'rb') as csvfile:
		proteinsReader = csv.reader(csvfile, delimiter=sep)
		proteinsReader.next(); #We ignore the first line, which is the title line
		for row in proteinsReader:
			geneName = row[0];
			belongings = [];
			for col in row[readStart:len(row)-1:]:
				belongings.append(col)
			expre[geneName] = np.array(belongings,dtype='int32');

	print('READER:: '+str(len(expre.keys()))+' rows read')
	m = len(expre.keys());
	n = len(expre.values()[0]);
	QC = np.zeros([m,n],dtype='int32')
	C  = np.zeros([m,n],dtype='int8')
	x = 0;
	for g in expre.keys():
		QC[x,:] = expre[g];
		C[x,np.nonzero(expre[g]>=thr)[0]]=1;
		x += 1;

	return C,QC,expre.keys()

def load_TaqMan_csv(fileName,thr=-1):
	D = np.loadtxt(fileName);
	U = np.unique(D);
	U.sort();
	zeroVal = -1;
	zeroFreq = 0;
	for u in U:
		numThisU = len(np.nonzero(D==u)[0]);
		if numThisU >= zeroFreq:
			zeroFreq = numThisU;
			zeroVal  = u;

	print("LOADING GEO FILE: "+fileName +"range: "+str(D.min())+"->"+str(D.max())+"zero val = "+str(zeroVal))
	C = np.ones(D.shape);
	if thr==-1:
		C[np.nonzero(D==zeroVal)]=0;
	else:
		print("threshold: "+str(thr))
		C[np.nonzero(D>=thr)]=0;

	Q = np.exp((-1.0/3.5)*(D-zeroVal))-1;


	if C.shape[0] < C.shape[1]:
		C = C.T;

	sumsC = C.sum(axis=1);
	C=C[np.nonzero(sumsC>0)[0],:]

	Q=Q[np.nonzero(sumsC>0)[0],:]

	return C,Q;
