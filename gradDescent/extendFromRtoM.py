'''
Author:          Victor Mireles - Freie Universitaet Berlin
Email:           syats.vm@gmail.com

Latest version:  https://github.com/syats/ModuleReusability

Distributed under GPL V3.0
'''


import numpy as np;
import decomposableMatrix as DM;
import aux.auxPreprocessing as app

#B is a decomposableMatrix of size r x r, all of whose sizes are 1.
#Returned is a list of decomposableMatrix of sizes from r+1 to m each with a smaller norm than the previous.

def regenerateTranslation(weightVector):
	m = int(weightVector.sum())
	translation = np.zeros(m,dtype='int');
	initial = 0;
	for i in range(len(weightVector)):
		w = weightVector[i];
		translation[initial:initial+w] = i;
		initial = initial + w;

	return translation;



def expandMatrix(B,translationOfRows=[]):
	if translationOfRows == []:
		trans = regenerateTranslation(B.weightVector)
	else:
		trans = translationOfRows
	return trans,DM.decomposableMatrix(app.uncleanMatrix(B.theMatrix.todense(),trans))
