

class AttributeTracking:
    def __init__(self,model):
        self.percent = 0
        self.probabilityList = []
        self.attAccuracySums = [[0]*model.env.formatData.numAttributes for i in range(model.env.formatData.numTrainInstances)]