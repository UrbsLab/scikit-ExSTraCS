import random
import copy

class AttributeTracking:
    def __init__(self,model):
        self.percent = 0
        self.probabilityList = []
        self.attAccuracySums = [[0]*model.env.formatData.numAttributes for i in range(model.env.formatData.numTrainInstances)]

    def updateAttTrack(self,model,pop):
        dataRef = model.env.dataRef
        if model.attribute_tracking_method == 'add':
            for ref in pop.correctSet:
                for each in pop.popSet[ref].specifiedAttList:
                    self.attAccuracySums[dataRef][each] += pop.popSet[ref].accuracy
        elif model.attribute_tracking_method == 'wh':
            tempAttTrack = [0]*model.env.formatData.numAttributes
            for ref in pop.correctSet:
                for each in pop.popSet[ref].specifiedAttList:
                    tempAttTrack[each] += pop.popSet[ref].accuracy
            for attribute_index in range(len(tempAttTrack)):
                self.attAccuracySums[dataRef][attribute_index] += model.attribute_tracking_beta * (tempAttTrack[attribute_index] - self.attAccuracySums[dataRef][attribute_index])

    def updatePercent(self, model):
        """ Determines the frequency with which attribute feedback is applied within the GA.  """
        self.percent = model.iterationCount/model.learning_iterations

    def getTrackProb(self):
        """ Returns the tracking probability list. """
        return self.probabilityList

    def genTrackProb(self,model):
        """ Calculate and return the attribute probabilities based on the attribute tracking scores. """
        #Choose a random data instance attribute tracking scores
        currentInstance = random.randint(0,model.env.formatData.numTrainInstances-1)
        #Get data set reference.
        trackList = copy.deepcopy(self.attAccuracySums[currentInstance])
        #----------------------------------------
        minVal = min(trackList)
        for i in range(len(trackList)):
            trackList[i] = trackList[i] - minVal
        maxVal = max(trackList)
        #----------------------------------------
        probList = []
        for i in range(model.env.formatData.numAttributes):
            if maxVal == 0.0:
                probList.append(0.5)
            else:
                probList.append(trackList[i]/float(maxVal + maxVal*0.01))  #perhaps make this float a constant, or think of better way to do this.

        self.probabilityList = probList

    def getSumGlobalAttTrack(self,model):
        """ For each attribute, sum the attribute tracking scores over all instances. For Reporting and Debugging"""
        globalAttTrack = [0.0 for i in range(model.env.formatData.numAttributes)]
        for i in range(model.env.formatData.numAttributes):
            for j in range(model.env.formatData.numTrainInstances):
                globalAttTrack[i] += self.attAccuracySums[j][i]
        return globalAttTrack