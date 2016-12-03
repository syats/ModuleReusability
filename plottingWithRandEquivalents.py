'''
Author:          Victor Mireles - Freie Universitaet Berlin
Email:           syats.vm@gmail.com

Latest version:  https://github.com/syats/ModuleReusability

Distributed under GPL V3.0
'''



import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib
from scipy.stats import binom
import aux.auxPreprocessing as app


#For each
def plotPerMatrixSeveralCurves(allStores,namesOfTypes=['REAL','DP-Rand','RSS-Rand','RSS2'],namesOfData=[],numPerType={'REAL':1,'RAND':1,'RSS3':1}):
	fileNames = allStores.keys()
	for fn in fileNames:
		plt.figure()


def plotWithReasAsBaes(allData,typesOfData,Cs,namesOfData=[],namesOfTypes=['REAL','DP-Rand','RSS-Rand','RSS2'],randToConsiderForSums=[1,2],markersOfTypes=['*','o','v','o'],colorsOfTypes=['r','b','g','k']):

	dataForViolinPlot = dict();

	portion = 1.5
	curveXLabel = "$log_{10}(k)$"
	matplotlib.rcParams.update({'font.size': 16})
	fileNames = allData.keys();
	sprows = 1;
	spcols = 1;
	if len(Cs)>1:
		sprows = 2;
		spcols = int(np.ceil(len(Cs)/2))+1;

	fig2 = plt.figure()
	subpnx = 0;
	subpny = 0;
	s=-1;
	for fn in fileNames:
		m = max(Cs[fn].shape)
		n = min(Cs[fn].shape)
		translationOfRows,C2 = app.cleanInputMatrix(Cs[fn])
		rr = min(C2.shape)
		print("\n"+str(Cs[fn].shape)+" "+ namesOfData[fn])
		s+=1;
		dataForViolinThisFN = [[],[],[]]
		placeYLabel = False
		if subpny==0:
			placeYLabel = True;
		placedLegends = [];
		placedLines   = [];
		if (len(fileNames)%spcols>0) and (s == 0):
			ax = plt.subplot2grid((sprows,spcols),(subpnx,subpny),colspan=2)
			subpny += 1;
		else:
			ax = plt.subplot2grid((sprows,spcols),(subpnx,subpny))
		subpny += 1;
		if subpny == spcols:
			subpny = 0;
			subpnx += 1;
		nextData = 1;

		#First we find the real data.
		for dataNum in range(len(allData[fn])):
			dataType = typesOfData[fn][dataNum];
			if dataType == 0:
				realData = allData[fn][dataNum];
				break
			else:
				continue

		for dataNum in range(len(allData[fn])):
			
			data     = allData[fn][dataNum];
			dataType = typesOfData[fn][dataNum];
			if dataType == 0:
				continue
			else:
				extraStyle=markersOfTypes[dataType]+''

			data = data[:len(realData)]
			#dataForViolinThisFN[dataType].append((data - realData).mean()/(Cs[fn].shape[1]*Cs[fn].shape[0]))
			ks = np.array(range(n,n+len(realData)))

			dfV = data.sum()/realData.sum();
			dataForViolinThisFN[dataType].append(dfV)
			#dataForViolinThisFN[dataType].append(((realData-data).sum()/realData.sum()))
			data = (realData - data)
			#print("lenData "+str(len(data))+" lenRealData"+str(len(realData)))

			X = range(Cs[fn].shape[1],int(len(data)/portion))
			Y = data[X];
			#Y = Y/X;
			X = np.array(X)
			X = np.log10(X);


			if not (dataType in placedLegends):
				ladded = ax.plot(X,Y,colorsOfTypes[dataType]+extraStyle,label=namesOfTypes[dataType])
				placedLines.append(ladded);
				placedLegends.append(dataType);
			else:
				ax.plot(X,Y,colorsOfTypes[dataType]+extraStyle)

		totalMin = 0;
		totalMax = 0.20
		#ax.set_ylim([0.98*totalMin,1.02*totalMax])

		ax.set_xlim([X.min() , X.max()])
		nbins = len(ax.get_xticklabels()) # added
		#ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='both'))

		if (len(fileNames) == 1):
			ax.legend(title=namesOfData[fn],numpoints=1)
		else:
			ax.legend([],title=namesOfData[fn],numpoints=1)

		ax.set_xlabel(curveXLabel)
		if placeYLabel:
			ax.set_ylabel("$\\rho(B_k(R)) - \\rho(B_k(C))$ ")
		#else:
		#	ax.set_yticklabels([]);

		tickPlaces = ax.get_xticks();
		xrangeV = [np.floor(X.min()) + 0.1*i for i in range(int(20*(X.max()-X.min())))]
		xrangeV = [xt for xt in xrangeV if (xt >= X.min()) and (xt <= X.max())]
		xrangeV2 = [xt for xt in xrangeV[1:-1] if ((10*xt % 5) == 0)]
		#xrangeV2.append(xrangeV[0])
		ax.set_xticks(xrangeV2)
		nbins = len(ax.get_xticklabels()) # added
		#ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='both'))

		dataForViolinPlot[namesOfData[fn]+"\n("+str(n)+")"] = dataForViolinThisFN;

	#Final adjustments
	if (len(fileNames) > 1):
		h, l = ax.get_legend_handles_labels()
		#fig2.legend(loc=1,ncol=1, borderaxespad=0.,labels=namesOfTypes[1:],handles=h[:len(namesOfTypes)],numpoints=1)
		fig2.subplots_adjust(wspace=0,hspace=0.25,left=0.06,right=0.98,top=0.93,bottom=0.11)
	else:
		fig2.subplots_adjust(wspace=0,hspace=0.15,left=0.24,right=0.80,top=0.97,bottom=0.12)

	h, l = ax.get_legend_handles_labels()
	fig2.legend(loc=9,ncol=2, borderaxespad=0.,labels=l,handles=h,numpoints=1)

	#plt.show()
	plt.close(fig2)

	return dataForViolinPlot


def makeViolinPlots(dataForViolinO,theAx,ytitle):

	dataForViolin = dict()
	for ke in dataForViolinO.keys():
		#if ke != 'GSE47652':
		#	continue
		dataForViolin[ke] = dataForViolinO[ke];


	ax = theAx

	titles = dataForViolin.keys();
	allData = []
	numFns = len(titles);
	for ke in titles:
		dataRAND = dataForViolin[ke][1];
		if np.var(dataRAND)<0.000001:
			dataRAND[0] = dataRAND[1]+0.0001;
		dataRSS  = dataForViolin[ke][2];
		if np.var(dataRSS)<0.000001:
			dataRSS[0] = dataRSS[1]+0.0001;
		allData.append(dataRAND)
		allData.append(dataRSS)

	positions = [[0.7*i,0.7*i] for i in range(1,numFns+1)];
	positions = [item for sublist in positions for item in sublist]
	ax.plot([positions[0]-1.05,positions[-1]+0.35],[1,1],'r:',alpha=0.7,zorder=-1)


	violin_parts = ax.violinplot(allData,showmeans=True,positions=positions,showextrema=False)
	#print(str(violin_parts['cmeans'].get_segments()))
	count = 0;
	Osegs = violin_parts['cmeans'].get_segments();
	segs=[]
	for b in violin_parts['bodies']:
		#b.set_alpha(0.75)
		thisSeg = Osegs[count];
		midPointX = thisSeg[0][0]+(thisSeg[1][0]-thisSeg[0][0])/2.0;
		if count % 2 == 0:
		    m = np.mean(b.get_paths()[0].vertices[:, 0])
		    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
		    b.set_color('b')
		    b.set_facecolor('blue')
		    newSeg = np.array([[thisSeg[0][0],thisSeg[0][1]]  ,  [ midPointX+0.01, thisSeg[1][1]]  ])
		else:
		    m = np.mean(b.get_paths()[0].vertices[:, 0])
		    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
		    b.set_color('g')
		    b.set_facecolor('green')
		    newSeg = np.array([[midPointX-0.01,thisSeg[0][1]]  ,  [ thisSeg[1][0], thisSeg[1][1]] ])
		count += 1;
		segs.append(newSeg);
		b.set_alpha(0.99)

	violin_parts['cmeans'].set_color('black')
	#print("seg:")

	violin_parts['cmeans'].set_segments(segs)

	#print(str(violin_parts['cmeans'].get_segments()))
	#print("__\n")


	ax.set_xticks([0.7*i for i in range(1,numFns+1)])
	ax.set_xticklabels(titles,rotation='vertical')
	ax.set_ylabel(ytitle)

	ylim = list(ax.get_ylim())
	if ylim[0]<1 and ylim[1] <= 1:
		ylim[1] = 1.1;
	if ylim[0] >= 1 and ylim[1] > 1:
		ylim[0] = 0.9
	ax.set_ylim(ylim)
	ax.set_yticks(ax.get_yticks()[1:-1])


	#Custom legend
	import matplotlib.patches as mpatches
	pB = mpatches.Patch(color='blue', linewidth=0)
	pG = mpatches.Patch(color='green', linewidth=0)
	#ax.legend((pB,pG),('DP-Rand / Real','RSS-Rand / Real'))






#allData is a dictionary with fileNames as keys and lists of vectors as elements.
#typesOfData is a dictionary with fileNames as keys and lists of integers as values
#Both dictionaries have the same number of elements, and the elements themselvs (which are lists) are also of the same length
#    for all i  len(allData.values[i]) == len(typesOfData.values[i])
#if dataI and typeI are the i'th elements of allData and typesOfData respectively, then dataI[j] is a vector that is to be plotted, and typeI[j] (which is an integer) represents the type of the vector. Types have associated a name (given by namesOfTypes, will be set in the legend) and a marker (given by markersOfTypes, usef for plotting style).
def makePlots(allData,typesOfData,Cs,namesOfData=[],namesOfTypes=['REAL','DP-Rand','RSS-Rand','RSS2'],randToConsiderForSums=[1,2],markersOfTypes=['*','o','v','o'],colorsOfTypes=['r','b','g','k'],fig1FName='',fig2FName='',plotKvsData=True,plotDataSums=True,sprows=2,spcols=4,plotTheHistograms=True,avgY=False,logX=True,logY=False,QCs=[]):
	fileNames = allData.keys();
	totalMax = max([max([deco.max() for deco in fnl]) for fnl in allData.values()])
	totalMin = min([min([deco.min() for deco in fnl]) for fnl in allData.values()])
	print("totals "+str(totalMin)+ " -> "+str(totalMax))


	portion = 2;
	matplotlib.rcParams.update({'font.size': 15})

	curveXLabel = "$k$, number of modules"
	if logX:
		curveXLabel = "$log_{10}(k)$, no. of modules"

	if avgY:
		minX = min([cc.shape[1] for cc in Cs.values()]);
		totalMax = max([ max( [ cur.max() for cur in allData[fn]]   )/(min(Cs[fn].shape)*max(Cs[fn].shape))   for fn in fileNames])

	if logY:
		totalMax = np.log10(totalMax);
		totalMin = np.log10(totalMin);

	print(str(totalMin)+"->"+str(totalMax))
	if len(namesOfData)==0:
		namesOfData = fileNames;


	#SUMS PLOT ------------------------------------------------
	if (plotDataSums) or (not(fig1FName=='')):
		fig = plt.figure()
		dist = 2;
		s = -1;
		xticks = [];
		xticklabels = [];
		for fn in fileNames:
			s += 1;
			X = dist+(dist*s)
			#print(str(Cs[fn].shape)+" ---------- 		")
			for randTC in randToConsiderForSums:
				realData = np.zeros(allData[fn][0].shape);
				meanRand = np.zeros(allData[fn][0].shape);
				meanRand = meanRand[:int(len(meanRand)/portion)]
				numRand  = 0.0;
				for dataNum in range(len(allData[fn])):
					data     = allData[fn][dataNum];
					noMoreChange = np.nonzero(data==Cs[fn].shape[0])[0]
					#print("r: "+str(noMoreChange[0]))
					data     = data[:len(meanRand)]
					if avgY:
						data = data/(Cs[fn].shape[0]*Cs[fn].shape[1])
					dataType = typesOfData[fn][dataNum]
					if dataType == randTC:
						meanRand += data;
						numRand  += 1;
					if dataType == 0:
						realData = data;


				meanRand = meanRand / numRand;
				sqdiff   = np.sqrt(np.power(meanRand-realData,2))

				if s>0:
					plt.plot(X,(sqdiff.mean()),colorsOfTypes[randTC]+markersOfTypes[randTC])
				else:
					lab = namesOfTypes[0]+" Vs. avg("+namesOfTypes[randTC]+")";
					plt.plot(X,(sqdiff.mean()),colorsOfTypes[randTC]+markersOfTypes[randTC],label=lab)

			xticks.append(X);
			xticklabels.append(namesOfData[fn]);




		plt.xlim([0,dist+(dist*(s+1))])
		fig.get_axes()[0].set_xticks(xticks);
		fig.get_axes()[0].set_xticklabels(xticklabels,rotation='vertical');
		fig.get_axes()[0].legend(numpoints=1)
		fig.get_axes()[0].set_ylabel("Sum of differences in densties")
		fig.get_axes()[0].set_xlabel("Data set")
		fig.subplots_adjust(bottom=0.30)

		if not(fig1FName==''):
			plt.savefig(fig1FName)
			plt.close()




	#CURVES PLOT -----------------------------------------------

	if plotKvsData or (not(fig2FName=='')):
		fig2 = plt.figure()
		subpnx = 0;
		subpny = 0;
		#fig2, axes2 = plt.subplots(nrows=sprows, ncols=spcols,sharey=True)
		s=-1;
		for fn in fileNames:
			m = max(Cs[fn].shape)
			print(m)
			s+=1;

			placeYLabel = False
			if subpny==0:
				placeYLabel = True;

			placedLegends = [];
			placedLines   = [];
			if (len(fileNames)%spcols>0) and (s == 0):
				ax = plt.subplot2grid((sprows,spcols),(subpnx,subpny),colspan=2)
				subpny += 1;
			else:
				ax = plt.subplot2grid((sprows,spcols),(subpnx,subpny))
			'''
			if (sprows > 1) and (spcols > 1):
				ax = axes2[subpnx,subpny];
			if (sprows > 1) and (spcols <= 1):
				ax = axes2[subpnx];
			if (sprows <= 1) and (spcols > 1):
				ax = axes2[subpny];
			if (sprows <= 1) and (spcols <= 1):
				ax = axes2
			print(str(fn)+" "+str([subpnx,subpny]))
			'''
			subpny += 1;
			if subpny == spcols:
				subpny = 0;
				subpnx += 1;

			nextData = 1;

			for dataNum in range(len(allData[fn])):
				data     = allData[fn][dataNum];
				dataType = typesOfData[fn][dataNum];
				if dataType == 0:
					extraStyle=markersOfTypes[dataType]+'-'
				else:
					extraStyle=markersOfTypes[dataType]+''

				X = range(Cs[fn].shape[1],int(len(data)/portion))
				Y = data[X];
				if avgY:
					Y = Y/(X);
					Y = Y/(float(m))
				if logX:
					X = np.log10(X);
				if logY:
					Y = np.log10(Y);
					totalMax = np.log10(totalMax)
					totalMin = np.log10(totalMin)

				if not (dataType in placedLegends):
					ladded = ax.plot(X,Y,colorsOfTypes[dataType]+extraStyle,label=namesOfTypes[dataType])
					placedLines.append(ladded);
					placedLegends.append(dataType);
				else:
					ax.plot(X,Y,colorsOfTypes[dataType]+extraStyle)

			if (len(X) == 0) or (X.max() == X.min()):
				continue
			ax.set_ylim([0.98*totalMin,1.02*totalMax])
			ax.set_xlim([X.min() , X.max()])
			nbins = len(ax.get_xticklabels()) # added
			#ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='both'))

			if (len(fileNames) == 1):
				ax.legend(title=namesOfData[fn],numpoints=1)
			else:
				ax.legend([],title=namesOfData[fn],numpoints=1)

			ax.set_xlabel(curveXLabel)
			if placeYLabel:
				ax.set_ylabel("Density of $B_k$")
			else:
				ax.set_yticklabels([]);

		#Final adjustments
		if (len(fileNames) > 1):
			h, l = ax.get_legend_handles_labels()
			fig2.legend(loc=1,ncol=1, borderaxespad=0.,labels=namesOfTypes,handles=h[:len(namesOfTypes)],numpoints=1)
			fig2.subplots_adjust(wspace=0,hspace=0.25,left=0.065,right=0.83,top=0.97,bottom=0.11)
		else:
			fig2.subplots_adjust(wspace=0,hspace=0.15,left=0.24,right=0.80,top=0.97,bottom=0.12)



	# Rowsum distribution histograms ----------------------------------------------------------------
	if plotTheHistograms:
		subpnx = 0;
		subpny = 0;
		fig3, axes3 = plt.subplots(nrows=sprows, ncols=spcols,sharey=True)
		fnnNum = -1;
		for fn in fileNames:
			fnnNum += 1;
			if (sprows > 1) and (spcols > 1):
				ax = axes3[subpnx,subpny];
			if (sprows > 1) and (spcols <= 1):
				ax = axes3[subpnx];
			if (sprows <= 1) and (spcols > 1):
				ax = axes3[subpny];
			if (sprows <= 1) and (spcols <= 1):
				ax = axes3
			print(str(fn)+" "+str([subpnx,subpny]))
			placeYLabel = False
			if subpny==0:
				placeYLabel = True;
			placeXLabel = False
			if subpnx>0:
				placeXLabel = True;

			subpny += 1;
			if subpny == spcols:
				subpny = 0;
				subpnx += 1;
			thisC = Cs[fn]
			toHist = thisC.sum(axis=1) / float(min(thisC.shape));
			weights = np.ones_like(toHist)/len(toHist)
			#print(str(toHist))

			if len(QCs)==len(Cs):
				thisQ = QCs[fn];
				Qsum = thisQ.sum(axis=1);
				toHist2 = thisQ.sum(axis=1) / Qsum.max();
				weights2 = np.ones_like(toHist2)/len(toHist2)
				hh2 = ax.hist(toHist,weights=weights,bins=10,alpha=0.9,label='Observed')
				#hh1 = ax.hist(toHist2,weights=weights2,bins=20,alpha=0.7,label='Normalized Sum of Number of Copies');
				#ax.text(0.3,0.9,namesOfData[fn])

			else:
				ax.hist(toHist,weights=weights,bins=10,label='Observed');

				#ax.text(0.3,0.6,namesOfData[fn])

			#Here we compute the histogram of a binomial distribution with the same density as C
			rho = thisC.sum()/float(thisC.shape[0]*thisC.shape[1]);
			bino = binom(min(thisC.shape),rho);
			binDistY = [bino.pmf(i) for i in range(min(thisC.shape)+1)]
			binDistX = [float(i)/float(min(thisC.shape)) for i in range(min(thisC.shape)+1)]
			ax.plot(binDistX,binDistY,'r-',label='Expected')



			ax.set_ylim([0,1])
			ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
			ax.set_xticklabels(['0','0.2','0.4','0.6','0.8','1'])
			if placeXLabel:
				ax.set_xlabel("miRNA sequence usage")
			if placeYLabel:
				ax.set_ylabel("Fraction of \nall miRNA Sequences")
			ax.legend([],title=namesOfData[fn])


			#ax.set_title()

		h, l = ax.get_legend_handles_labels()
		fig3.legend(loc=9,ncol=2, borderaxespad=0.,labels=l,handles=h)

		fig3.subplots_adjust(wspace=0.079,hspace=0.13,left=0.09,right=0.99,top=0.92,bottom=0.13)




		plt.show()








# --------- B A N D S ---------
def makeBands(allData,typesOfData,Cs,namesOfData=[],namesOfTypes=['REAL','DP-Rand','RSS-Rand','RSS2'],randToConsiderForSums=[1,2],markersOfTypes=['*','o','v','o'],colorsOfTypes=['r','b','g','k'],fig1FName='',fig2FName='',plotKvsData=True,plotDataSums=True,avgY=False,logX=False,logY=False,QCs=[],numStds = 2,theAx=None,yLabel=None,legendPos=0,finalK=None,lightColors=['#ff6666','#6666ff','#66ff66']):
	fileNames = allData.keys();
	totalMax = max([max([deco.max() for deco in fnl]) for fnl in allData.values()])
	totalMin = min([min([deco.min() for deco in fnl]) for fnl in allData.values()])
	print("totals "+str(totalMin)+ " -> "+str(totalMax))


	portion = 1;


	curveXLabel = "$k$ (number of modules)"
	if logX:
		curveXLabel = "$log_{10}(k)$ (number of modules)"

	if avgY:
		minX = min([cc.shape[1] for cc in Cs.values()]);
		totalMax =  max([ max( [ cur.max() for cur in allData[fn]]   )/(min(Cs[fn].shape)*max(Cs[fn].shape))   for fn in fileNames])

	if logY:
		totalMax = np.log10(totalMax);
		totalMin = np.log10(totalMin);

	print(str(totalMin)+"->"+str(totalMax))
	if len(namesOfData)==0:
		namesOfData = fileNames;


	#-- B A N D S    PLOT -----------------------------------<

	if plotKvsData or (not(fig2FName=='')):

		subpnx = 0;
		subpny = 0;
		#fig2, axes2 = plt.subplots(nrows=sprows, ncols=spcols,sharey=True)
		s=-1;
		for fn in fileNames:
			m = max(Cs[fn].shape)
			print(str(m)+" "+str(len(allData[fn])))
			s+=1;

			placeYLabel = False
			if subpny==0:
				placeYLabel = True;

			placedLegends = [];
			placedLines   = [];


			ax = theAx


			nextData = 1;

			bands = [dict() for ii in range(max(typesOfData[fn])+1 )]
			for band in bands:
				for x in range(m+1):
					band[x]  = []

			for dataNum in range(len(allData[fn])):
				data     = allData[fn][dataNum];
				dataType = typesOfData[fn][dataNum];

				if dataType == 0:
					extraStyle=markersOfTypes[dataType]+'-'
				else:
					extraStyle=markersOfTypes[dataType]+''

				if finalK == None:
					X = range(Cs[fn].shape[1],int(len(data)/portion))
				else:
					X = range(Cs[fn].shape[1],finalK)


				if (data[0]==0) and (data[1]==0):
					Y = data[X];
				else:
					Y = data;


				if logY:
					Y = np.log10(Y);
					totalMax = np.log10(totalMax)
					totalMin = np.log10(totalMin)

				print("minband "+str(min(Y))+" << "+str(min(data))+" - "+str(max(data)))
				for kn in range(len(X)):
					x = X[kn];
					y = Y[kn];
					if avgY:
						bands[dataType][x].append(y/x)
					else:
						bands[dataType][x].append(y)


			for ii in range(max(typesOfData[fn])+1):

				if ii == 0:
					extraStyle=markersOfTypes[ii]+'-'
				else:
					extraStyle=markersOfTypes[ii]+''
				print("dt "+str(ii)+" > "+colorsOfTypes[ii]+extraStyle)
				dataType = namesOfTypes[ii];
				band = bands[ii];

				XX = [];
				YY1a = [];
				YY   = []
				YY2a = [];
				YY1b = [];
				YY2b = [];
				for x in band.keys():
					if len(band[x])<1:
						continue
					xp = x;
					minim = np.mean(band[x])+numStds*np.std(band[x])
					maxim = np.mean(band[x])-numStds*np.std(band[x])
					if logX:
						xp = np.log10(x);
					XX.append(xp)
					YY1a.append(minim)
					YY2a.append(maxim)
					thisBand = band[x];
					thisBand.sort()
					if len(thisBand) > 1:
						YY1b.append(thisBand[1])
						YY2b.append(thisBand[-2])
					else:
						YY1b.append(thisBand[0])
						YY2b.append(thisBand[0])
					YY.append(np.mean(band[x]))
				YY1 = YY1b;
				YY2 = YY2b;
				print("YY1 "+str(min(YY1))+"  ->  "+str(max(YY1)))


				if not (dataType in placedLegends):
					if (ii==0):
						ladded=ax.plot(XX,YY1,colorsOfTypes[ii]+extraStyle,label=namesOfTypes[ii],markeredgewidth=0.0,zorder=100,fillstyle='full',linewidth=2)
						print("yy1min"+str(min(YY1)));
						placedLines.append(ladded[0]);
					else:
						#STDs
						ax.fill_between(XX,YY1a,YY2a,facecolor=colorsOfTypes[ii],edgecolor=colorsOfTypes[ii],zorder=20,interpolate=False,label=namesOfTypes[ii])
						ladded=plt.Rectangle((0, 0), 1, 1, fc=colorsOfTypes[ii])
						#MIN MAX
						ax.fill_between(XX,YY1b,YY2b,facecolor=lightColors[ii],edgecolor=lightColors[ii],zorder=1,interpolate=False)
						ladded=plt.Rectangle((0, 0), 1, 1, fc=colorsOfTypes[ii])
						placedLines.append(ladded);
						#mean
						#ladded=ax.plot(XX,YY,colorsOfTypes[ii])
					placedLegends.append(ii);

				else:
					if (ii==0):
						ladded=ax.plot(XX,YY1,colorsOfTypes[ii]+extraStyle)
						print("yy1min"+str(min(YY1)));
					else:
						ladded=ax.fill_between(XX,YY1,YY2,facecolor=colorsOfTypes[ii],edgecolor=colorsOfTypes[ii],alpha=0.8,interpolate=False,label=namesOfTypes[dataType])






			#ax.set_ylim([0.98*totalMin,1.02*totalMax])
			#ax.set_xlim([min(X) , max(X)])
			nbins = len(ax.get_xticklabels()) # added
			ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='both'))

			ylabs = ax.get_yticks();
			ax.set_yticks(ylabs[1:-1])

			if not (legendPos==None):
				ax.legend(placedLines,[namesOfData[fn]]+namesOfTypes[1:],numpoints=1,loc=legendPos)
				ax.set_xticks([])

			ax.set_xlabel(curveXLabel)
			if placeYLabel:
				ax.set_ylabel(yLabel)
			else:
				ax.set_yticklabels([]);
