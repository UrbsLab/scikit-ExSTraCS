
from skExSTraCS import ExSTraCS
from skrebate import ReliefF
import pandas as pd
import numpy as np
from skExSTraCS import StringEnumerator
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from skeLCS import eLCS

converter = StringEnumerator('test/DataSets/Real/ContAndMissing.csv','panc_type01')
converter.deleteAttribute("plco_id")
headers,classLabel,dataFeatures,dataPhenotypes = converter.getParams()

formatted = np.insert(dataFeatures,dataFeatures.shape[1],dataPhenotypes,1)
np.random.shuffle(formatted)
dataFeatures = np.delete(formatted,-1,axis=1)
dataPhenotypes = formatted[:,-1]

# for i in range(len(dataFeatures)):
#     for j in range(len(dataFeatures[0])):
#         if np.isnan(dataFeatures[i,j]):
#             dataFeatures[i,j] = 0
#
# clf = MLPClassifier()
# clf.fit(dataFeatures,dataPhenotypes)
# print(np.mean(cross_val_score(clf,dataFeatures,dataPhenotypes,cv=3)))
# print(clf.score(dataFeatures,dataPhenotypes))

rbSample = np.random.choice(formatted.shape[0],1000,replace=False)
newL = []
for i in rbSample:
    newL.append(formatted[i])
newL = np.array(newL)
dataFeaturesR = np.delete(newL,-1,axis=1)
dataPhenotypesR = newL[:,-1]

print("ReliefF begins")
r = ReliefF()
r.fit(dataFeaturesR,dataPhenotypesR)
scores = r.feature_importances_

print("Training Begins")
#model = eLCS(learningIterations=10000)
model = ExSTraCS(learningIterations=10000,expertKnowledge=scores,ruleCompaction=None) #PDRC/CRA2 makes minimal change, QRC/QRF do poorly
model.fit(dataFeatures,dataPhenotypes)

print("Evaluation Begins")
print(model.score(dataFeatures,dataPhenotypes))
#print(np.mean(cross_val_score(model,dataFeatures,dataPhenotypes,cv=3)))

print("Auxilliary Evaluation")
#b = model.predict_proba(dataFeatures)
#print(model.getFinalAttributeSpecificityList())
#print(model.getFinalTrainingAccuracy())
#print(model.getFinalAttributeAccuracyList())
#print(model.getFinalInstanceCoverage())
#print(model.getFinalAttributeCooccurences(headers))
#print(model.getFinalAttributeTrackingSums())

print("Export Begins")
# model.exportIterationTrackingData('defaultExportDir/tracking.csv')
# model.exportFinalRulePopulation(headers,classLabel,'defaultExportDir/popBeforeRC.csv',False,False)
#model.exportFinalRulePopulation(headers,classLabel,'defaultExportDir/popAfterRC.csv',True,True)
#
# model.pickleModel('defaultExportDir/pickle1',True)
# model.pickleModel('defaultExportDir/pickle1b',False)
#
# print("Training 2 Begins")
# model2 = ExSTraCS(learningIterations=5000,expertKnowledge=scores,N=2000,nu=10,trackAccuracyWhileFit=True,rebootFilename='defaultExportDir/pickle1')
# model2.fit(dataFeatures,dataPhenotypes)
# print(model2.score(dataFeatures,dataPhenotypes))
# model2.exportIterationTrackingData('defaultExportDir/tracking2.csv')
#
# print("Training 3 Begins")
# model3 = ExSTraCS(learningIterations=5000,expertKnowledge=scores,N=2000,nu=10,trackAccuracyWhileFit=True,rebootFilename='defaultExportDir/pickle1b')
# model3.fit(dataFeatures,dataPhenotypes)
# print(model3.score(dataFeatures,dataPhenotypes))
# model3.exportIterationTrackingData('defaultExportDir/tracking2b.csv')
