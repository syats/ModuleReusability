'''
Author:          Victor Mireles - Freie Universitaet Berlin
Email:           syats.vm@gmail.com

Latest version:  https://github.com/syats/ModuleReusability

Distributed under GPL V3.0
'''


import networkx as nx
import numpy as np


#Given a list of sets B, returns the union of all of them
def unionOfSets(B):
	result = set();
	for b in B:
		result = result | set(list(b));
	return result


#Given a graph and binary node weights for the set of vertices, it constructs a set of graphs {G1,G2,...,Gn} where Gj is the graph induced by all those vertices whose weight w has a 1 in the j'th entry
def graphsFromWeights(G,W):
	C=[];
	V = G.nodes();
	E = G.edges();

	ke=W.keys();
	protypeVector = W.get(ke[0]);
	numGraphsToOutput = protypeVector.shape[0];
	for gnum in range(numGraphsToOutput):
		g=nx.Graph();
		nodes = 0;
		for e in E:
			gene1Name = e[0];
			gene2Name = e[1];
			isGene1   = W.get(gene1Name)[gnum];
			isGene2   = W.get(gene2Name)[gnum];
			addGene1  = isGene1==1;
			addGene2  = isGene2==1;
			if (addGene1):
				g.add_node(gene1Name);
				nodes+=1;
			if (addGene2):
				g.add_node(gene2Name);
				nodes+=1;
			if ( (addGene1 & addGene2) ):
				g.add_edge(gene1Name,gene2Name);
		C.append(g.copy());
	return C

#G is the total network
#W tells us, for every gene, if it belongs in a condition or not
def conditionMatrixFromWeights(G,W,nodeNames):
	V = G.nodes();
	E = G.edges();

	ke=W.keys();
	protypeVector = W.get(ke[0]);
	n = protypeVector.shape[0];
	m = len(G.nodes())

	C=np.zeros([n,m]);

	for v in V:
		nodeName = nodeNames[int(v)]
		C[:,int(v)] = W.get(nodeName);

	return C
