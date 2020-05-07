
from skExSTraCS import ExSTraCS
from skrebate import ReliefF
import pandas as pd
import numpy as np
from skExSTraCS import StringEnumerator
from sklearn.model_selection import cross_val_score

converter = StringEnumerator('test/DataSets/Real/Multiplexer6.csv','class')
headers,classLabel,dataFeatures,dataPhenotypes = converter.getParams()

formatted = np.insert(dataFeatures,dataFeatures.shape[1],dataPhenotypes,1)
np.random.shuffle(formatted)
dataFeatures = np.delete(formatted,-1,axis=1)
dataPhenotypes = formatted[:,-1]

print("ReliefF begins")
r = ReliefF()
r.fit(dataFeatures,dataPhenotypes)
scores = r.feature_importances_

model = ExSTraCS(learningIterations=1000,N=1000,nu=10,expertKnowledge=[1,1,0,0,0,0])
model.fit(dataFeatures,dataPhenotypes)
print(model.score(dataFeatures,dataPhenotypes))
