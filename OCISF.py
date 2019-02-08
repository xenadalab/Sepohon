# Bismillahhirahmanirahim

import os
from datetime import datetime
from sklearn.externals import joblib
from sklearn.ensemble import IsolationForest

regressenname = 'ocisf'  # <--- Don't forget to update this

datasetpath = '/casawaridisk/Share/Dataset/Tautau/autoesom-htautau.pkl'
outputdir = '/casawaridisk/qunox/ML/Sepohon/Output/Sepohon_Analysis'   # <--- Change to your dir
outputfilename = 'ocisf_%s.pkl' % datetime.now().strftime('%Y%m%d%H%M%S')
lowfeals = ['LL0', 'LL1', 'LL2', 'LL3', 'LL4', 'LL5', 'LL6', 'LL7', 'LL8', 'LL9']
highfeals = ['HL0', 'HL1', 'HL2', 'HL3', 'HL4', 'HL5', 'HL6', 'HL7', 'HL8', 'HL9', 'HL10', 'HL11',
             'HL12', 'HL13', 'HL14']

print('Start')
print('Loading the dataset')
dt = joblib.load(datasetpath)
trainnoi_1_df = dt['noise1_df']
trainnoi_2_df = dt['noise2_df']
test_df = dt['testdf']
trainnoi_df = trainnoi_1_df.append(trainnoi_2_df)
print('Fininsh Loading')

print('Creating the classifier model')
cls = IsolationForest(verbose=True, n_jobs=-1)

print('Training the function')
trainmat = trainnoi_df[highfeals].as_matrix()
cls.fit(trainmat)
print('Finish training')

print('Testing the classifier')
testmat = test_df[highfeals].as_matrix()
result = cls.predict(testmat)
scorels = cls.decision_function(testmat)
print('Done classifying')

print('Saving the output')
resultdic = {
                'predictionresult': result,
                'regressorname': regressenname,
                'cls':cls,
                'score': scorels,
                'datapath': datasetpath
            }

joblib.dump(resultdic, os.path.join(outputdir, outputfilename))
print('Alhamdulilah done')