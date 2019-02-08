# Bismillahhirahmanirahim

import os
from time import time
from numpy import mean, std
from datetime import datetime
from numpy.random import choice
from collections import Counter
from numpy import matrix, append, array
from sklearn.externals import joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestRegressor
# --------------------------------------------- Parameter --------------------------------------------------------------

datasetpath = '/casawaridisk/Share/Dataset/Tautau/autoesom-htautau.pkl'
outputdir = '/casawaridisk/qunox/ML/Sepohon/Output/Sepohon_Analysis'   # <--- Change to your dir
outputfilename = 'sepohonscrfhtautau_result_%s.pkl' % datetime.now().strftime('%Y%m%d%H%M%S')
lowfeals = ['LL0', 'LL1', 'LL2', 'LL3', 'LL4', 'LL5', 'LL6', 'LL7', 'LL8', 'LL9']
highfeals = ['HL0', 'HL1', 'HL2', 'HL3', 'HL4', 'HL5', 'HL6', 'HL7', 'HL8', 'HL9', 'HL10', 'HL11',
             'HL12', 'HL13', 'HL14']

maxcycle = 700
max_highfea_inuse = 5
clusternum = 50
regressionsamplesizeratio = 0.3

# -------------------------------------------------- Main --------------------------------------------------------------
print('Start')
print('Loading the dataset')
dt = joblib.load(datasetpath)
trainnoi_1_df = dt['noise1_df']
trainnoi_2_df = dt['noise2_df']
test_df = dt['testdf']
print('Fininsh Loading')

mainregresstrainsampleindex_ls = trainnoi_1_df.index.values
clustertrainsampleindex_ls = trainnoi_2_df.index.values
testingsampleindex_ls = test_df.index.values

totaltestingsample = len(testingsampleindex_ls)
totalsignal = len(test_df.loc[test_df['label'] == 1])
totalregsample = len(trainnoi_1_df)
totalclstrainsample = len(trainnoi_2_df)
highfeanum = len(highfeals)
regressionsamplesize = int(len(mainregresstrainsampleindex_ls) * regressionsamplesizeratio)
print('Starting the scoring cycle')

print('Creating the regression model')

regmodel = RandomForestRegressor(n_jobs=-1)

cyclels = []
feausels = []
ratiosls = []
meanls = []
stdls = []
cycleperiodls = []
samplescoremat = matrix([[0] for i in range(totaltestingsample)])

for cycle in range(maxcycle):
    print('\n\n\n==> Starting cycle: %s (%.2f)' % (cycle, cycle/maxcycle))
    starttime = time()
    print('==> Selecting the high feature')
    inuse_highfea_ls = choice(highfeals, size=max_highfea_inuse, replace=False)
    regresstrainsampleindex_ls = choice(mainregresstrainsampleindex_ls, size=regressionsamplesize, replace=False)
    print('==> High feature in use: %s' % ','.join(inuse_highfea_ls))

    # Creatinge the regression sample and model
    print('==> Creating regression matrix')
    regdf = trainnoi_1_df.loc[regresstrainsampleindex_ls]
    reg_lowfea_mat = regdf[lowfeals].values
    reg_highfea_mat = regdf[inuse_highfea_ls].values

    clstraindf = trainnoi_2_df.loc[clustertrainsampleindex_ls]
    clstrain_lowfea_mat = clstraindf[lowfeals].values
    clstrain_highfea_mat = clstraindf[inuse_highfea_ls].values

    inusetestsample_df = test_df.loc[testingsampleindex_ls]
    inusetestsample_lowfea_mat = inusetestsample_df[lowfeals].values
    trueinusetestsample_highfea_mat = inusetestsample_df[inuse_highfea_ls].values

    print('==> Training the regression model')
    regmodel.fit(reg_lowfea_mat, reg_highfea_mat)
    print('==> Done training')

    # Regressing the sample
    print('==> Regressing the sample and clstrain matrix')
    predict_inusetestsample_highfea_mat = regmodel.predict(inusetestsample_lowfea_mat)
    predict_clstrain_highfea_mat = regmodel.predict(clstrain_lowfea_mat)
    predict_reg_highfea_mat = regmodel.predict(reg_lowfea_mat)
    print('==> Finish Regressing the sample and clstrain matrix')

    print('==> Calculating the regression error')
    regerr_inusetestsample_mat = trueinusetestsample_highfea_mat - predict_inusetestsample_highfea_mat
    regerr_clstrain_mat = clstrain_highfea_mat - predict_clstrain_highfea_mat
    regerr_reg_mat = reg_highfea_mat - predict_reg_highfea_mat

    # Clustering the sample
    print('==> Creating the cluster model')
    clsmodel = MiniBatchKMeans(n_clusters=clusternum, max_iter=1000)  # <--- Cluster model
    print('==> Training the cluster model')
    clsmodel.fit(regerr_inusetestsample_mat)
    print('==> Done training the cluster model')

    print('==> Clustering the testing sample')
    clslabel_inusetestsample_ls = clsmodel.predict(regerr_inusetestsample_mat)
    clslabel_clstrain_ls = clsmodel.predict(regerr_clstrain_mat)

    print('==> Calculating cluster distributions')
    counter_inusetestsample_dict = Counter(clslabel_inusetestsample_ls)
    counter_clstrain_dict = Counter(clslabel_clstrain_ls)

    cycleclsratiols = []
    cycleclsratiodic = {}

    print('\n\nCycle-%s Result' % cycle)
    print('------------------------------------------------------------------------')
    print('Class\tTest\tTrain\tRatio')
    for i in counter_inusetestsample_dict.keys():
        clstraincount = counter_clstrain_dict[i]
        if clstraincount == 0: clstraincount = 1
        clsratio = counter_inusetestsample_dict[i] / clstraincount
        cycleclsratiols.append(clsratio)
        cycleclsratiodic[i] = clsratio
        txt = '%s\t%s\t%s\t%.3f' % (i,counter_inusetestsample_dict[i],clstraincount, clsratio)
        print(txt)
    print('------------------------------------------------------------------------\n\n')

    print('==> Scoring the sample')
    cyclesamplescorels = [cycleclsratiodic[i] for i in clslabel_inusetestsample_ls]
    cyclemean = mean(cyclesamplescorels)
    cyclestd = std(cyclesamplescorels)
    cyclesamplescorevec = array(cyclesamplescorels).reshape([-1, 1])
    samplescoremat = append(samplescoremat, cyclesamplescorevec, axis=1)

    print('==> Score mean: %.3f std:%.3f' % (cyclemean, cyclestd))
    print('==> End of Cycle')

    cyclels.append(cycle)
    feausels.append(inuse_highfea_ls)
    ratiosls.append(cycleclsratiols)
    meanls.append(cyclemean)
    stdls.append(cyclestd)
    cycletime = time() - starttime
    cycleperiodls.append(cycletime)

    print('==> Cycle period: %.3f second Eta: %.3f hours' % (cycletime, ((maxcycle - cycle) * cycletime) / 3600))

resultdic = {
                'datasetpath' : datasetpath,
                'cyclels' : cyclels,
                'feausels' : feausels,
                'mean': meanls,
                'std':stdls,
                'scoremat': samplescoremat,
                'processingtime' : cycleperiodls,
                'lowfea': lowfeals,
                'highfea': highfeals,
            }

joblib.dump(resultdic, os.path.join(outputdir, outputfilename))
print('Alhamdulilah done')
