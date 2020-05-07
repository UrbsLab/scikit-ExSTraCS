
from skExSTraCS import ExSTraCS
from skrebate import ReliefF
import pandas as pd
import numpy as np
from skExSTraCS import StringEnumerator
from sklearn.model_selection import cross_val_score

converter = StringEnumerator('test/DataSets/Real/Multiplexer20Modified.csv','Class')
headers,classLabel,dataFeatures,dataPhenotypes = converter.getParams()

# formatted = np.insert(dataFeatures,dataFeatures.shape[1],dataPhenotypes,1)
# np.random.shuffle(formatted)
# dataFeatures = np.delete(formatted,-1,axis=1)
# dataPhenotypes = formatted[:,-1]

print("ReliefF begins")
# r = ReliefF()
# r.fit(dataFeatures,dataPhenotypes)
# scores = r.feature_importances_
scores = [0.080835,0.071416,0.076315,0.074602,0.000877,-0.000606,0.003651,-0.002214,-0.000608,-0.002425,0.000013,0.00343,-0.001186,-0.001607,0.000061,-0.000367,0.001698,0.000787,0.001014,0.001723]

s = 0
for seed in range(30):
    print("Training Begins")
    model = ExSTraCS(learningIterations=5000,expertKnowledge=scores,N=2000,nu=10,trackAccuracyWhileFit=True,randomSeed=seed)
    model.fit(dataFeatures,dataPhenotypes)

    print("Evaluation Begins")
    s += model.score(dataFeatures,dataPhenotypes)

print(s/30)
# print(model.predict(dataFeatures))
# print(model.predict_proba(dataFeatures))
# print(model.getFinalAttributeSpecificityList())
# print(model.getFinalTrainingAccuracy())
# print(model.getFinalAttributeAccuracyList())
# print(model.getFinalInstanceCoverage())
# print(model.getFinalAttributeCooccurences(headers))
# print(model.getFinalAttributeTrackingSums())
#
# print("Export Begins")
# model.exportIterationTrackingData('defaultExportDir/tracking.csv')
# model.exportFinalRulePopulation(headers,classLabel,'defaultExportDir/popBeforeRC.csv',False,False)
# model.exportFinalRulePopulation(headers,classLabel,'defaultExportDir/popAfterRC.csv',True,True)
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
