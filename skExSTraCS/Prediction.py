import numpy as np
import random

class Prediction():
    def __init__(self,model,population):
        self.decision = None
        self.probabilities = {}
        self.hasMatch = len(population.matchSet) != 0

        #Discrete Phenotypes
        if model.env.formatData.discretePhenotype:
            self.vote = {}
            self.tieBreak_Numerosity = {}
            self.tieBreak_TimeStamp = {}

            for eachClass in model.env.formatData.phenotypeList:
                self.vote[eachClass] = 0.0
                self.tieBreak_Numerosity[eachClass] = 0.0
                self.tieBreak_TimeStamp[eachClass] = 0.0

            for ref in population.matchSet:
                cl = population.popSet[ref]
                self.vote[cl.phenotype] += cl.fitness * cl.numerosity * model.env.formatData.classPredictionWeights[cl.phenotype]
                self.tieBreak_Numerosity[cl.phenotype] += cl.numerosity
                self.tieBreak_TimeStamp[cl.phenotype] += cl.initTimeStamp

            #Populate Probabilities
            sProb = 0
            for k,v in sorted(self.vote.items()):
                self.probabilities[k] = v
                sProb += v
            if sProb == 0: #In the case the match set doesn't exist
                for k, v in sorted(self.probabilities.items()):
                    self.probabilities[k] = 0
            else:
                for k,v in sorted(self.probabilities.items()):
                    self.probabilities[k] = v/sProb

            highVal = 0.0
            bestClass = []
            for thisClass in model.env.formatData.phenotypeList:
                if self.vote[thisClass] >= highVal:
                    highVal = self.vote[thisClass]

            for thisClass in model.env.formatData.phenotypeList:
                if self.vote[thisClass] == highVal:  # Tie for best class
                    bestClass.append(thisClass)

            if highVal == 0.0:
                self.decision = None

            elif len(bestClass) > 1:
                bestNum = 0
                newBestClass = []
                for thisClass in bestClass:
                    if self.tieBreak_Numerosity[thisClass] >= bestNum:
                        bestNum = self.tieBreak_Numerosity[thisClass]

                for thisClass in bestClass:
                    if self.tieBreak_Numerosity[thisClass] == bestNum:
                        newBestClass.append(thisClass)

                if len(newBestClass) > 1:
                    bestStamp = 0
                    newestBestClass = []
                    for thisClass in newBestClass:
                        if self.tieBreak_TimeStamp[thisClass] >= bestStamp:
                            bestStamp = self.tieBreak_TimeStamp[thisClass]

                    for thisClass in newBestClass:
                        if self.tieBreak_TimeStamp[thisClass] == bestStamp:
                            newestBestClass.append(thisClass)
                    # -----------------------------------------------------------------------
                    if len(newestBestClass) > 1:  # Prediction is completely tied - extracs has no useful information for making a prediction
                        self.decision = 'Tie'
                else:
                    self.decision = newBestClass[0]
            else:
                self.decision = bestClass[0]

        if self.decision == None or self.decision == 'Tie':
            if model.env.formatData.discretePhenotype:
                self.decision = model.env.formatData.majorityClass
                #self.decision = random.choice(model.env.formatData.phenotypeList)
            else:
                self.decision = random.randrange(model.env.formatData.phenotypeList[0],model.env.formatData.phenotypeList[1],(model.env.formatData.phenotypeList[1]-model.env.formatData.phenotypeList[0])/float(1000))


    def getFitnessSum(self, population, low, high):
        """ Get the fitness sum of rules in the rule-set. For continuous phenotype prediction. """
        fitSum = 0
        for ref in population.matchSet:
            cl = population.popSet[ref][0]
            if cl.phenotype[0] <= low and cl.phenotype[1] >= high:  # if classifier range subsumes segment range.
                fitSum += cl.fitness
        return fitSum

    def getDecision(self):
        """ Returns prediction decision. """
        return self.decision

    def getProbabilities(self):
        ''' Returns probabilities of each phenotype from the decision'''
        a = np.empty(len(sorted(self.probabilities.items())))
        counter = 0
        for k, v in sorted(self.probabilities.items()):
            a[counter] = v
            counter += 1
        return a
