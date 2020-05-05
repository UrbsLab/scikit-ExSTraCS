import random

class Classifier:
    def __init__(self,model):
        self.specifiedAttList = []
        self.condition = []
        self.phenotype = None

        self.fitness = model.init_fitness
        self.accuracy = 0
        self.numerosity = 1
        self.aveMatchSetSize = None
        self.deletionProb = None

        self.timeStampGA = None
        self.initTimeStamp = None
        self.epochComplete = False

        self.matchCount = 0
        self.correctCount = 0
        self.matchCover = 0
        self.correctCover = 0

    def initializeByCovering(self,model,setSize,state,phenotype):
        self.timeStampGA = model.iterationCount
        self.initTimeStamp = model.iterationCount
        self.aveMatchSetSize = setSize
        self.phenotype = phenotype

        toSpecify = random.randint(1, model.ruleSpecificityLimit)
        if model.useExpertKnowledge:
            i = 0
            while len(self.specifiedAttList) < toSpecify and i < model.env.formatData.numAttributes - 1:
                target = model.EK.EKRank[i]
                if state[target] != None:
                    self.specifiedAttList.append(target)
                    self.condition.append(self.buildMatch(model,target,state))
                i += 1
        else:
            potentialSpec = random.sample(range(model.env.formatData.numAttributes),toSpecify)
            for attRef in potentialSpec:
                if state[attRef] == None:
                    self.specifiedAttList.append(attRef)
                    self.condition.append(self.buildMatch(model,attRef,state))

    def buildMatch(self,model,attRef,state):
        attributeInfoType = model.env.formatData.attributeInfoType[attRef]
        if not (attributeInfoType):  # Discrete
            attributeInfoValue = model.env.formatData.attributeInfoDiscrete[attRef]
        else:
            attributeInfoValue = model.env.formatData.attributeInfoContinuous[attRef]

        if attributeInfoType: #Continuous Attribute
            attRange = attributeInfoValue[1] - attributeInfoValue[0]
            rangeRadius = random.randint(25, 75) * 0.01 * attRange / 2.0  # Continuous initialization domain radius.
            Low = state[attRef] - rangeRadius
            High = state[attRef] + rangeRadius
            condList = [Low, High]
        else:
            condList = state[attRef]
        return condList

    def updateEpochStatus(self,model):
        if not self.epochComplete and (model.iterationCount - self.initTimeStamp - 1) >= model.env.formatData.numTrainInstances:
            self.epochComplete = True

    def match(self, model, state):
        for i in range(len(self.condition)):
            specifiedIndex = self.specifiedAttList[i]
            attributeInfoType = model.env.formatData.attributeInfoType[specifiedIndex]
            # Continuous
            if attributeInfoType:
                instanceValue = state[specifiedIndex]
                if instanceValue == None:
                    return False
                elif self.condition[i][0] < instanceValue < self.condition[i][1]:
                    pass
                else:
                    return False

            # Discrete
            else:
                stateRep = state[specifiedIndex]
                if stateRep == self.condition[i]:
                    pass
                elif stateRep == None:
                    return False
                else:
                    return False
        return True

    def equals(self,cl):
        if cl.phenotype == self.phenotype and len(cl.specifiedAttList) == len(self.specifiedAttList):
            clRefs = sorted(cl.specifiedAttList)
            selfRefs = sorted(self.specifiedAttList)
            if clRefs == selfRefs:
                for i in range(len(cl.specifiedAttList)):
                    tempIndex = self.specifiedAttList.index(cl.specifiedAttList[i])
                    if not (cl.condition[i] == self.condition[tempIndex]):
                        return False
                return True
        return False

    def updateExperience(self):
        self.matchCount += 1
        if self.epochComplete:  # Once epoch Completed, number of matches for a unique rule will not change, so do repeat calculation
            pass
        else:
            self.matchCover += 1

    def updateMatchSetSize(self, model,matchSetSize):
        if self.matchCount < 1.0 / model.beta:
            self.aveMatchSetSize = (self.aveMatchSetSize * (self.matchCount-1)+ matchSetSize) / float(self.matchCount)
        else:
            self.aveMatchSetSize = self.aveMatchSetSize + model.beta * (matchSetSize - self.aveMatchSetSize)

    def updateCorrect(self):
        self.correctCount += 1
        if self.epochComplete: #Once epoch Completed, number of correct for a unique rule will not change, so do repeat calculation
            pass
        else:
            self.correctCover += 1

    def updateAccuracy(self):
        self.accuracy = self.correctCount / float(self.matchCount)

    def updateFitness(self,model):
        self.fitness = pow(self.accuracy, model.nu)

    def updateNumerosity(self, num):
        """ Alters the numberosity of the classifier.  Notice that num can be negative! """
        self.numerosity += num

    def isSubsumer(self, model):
        if self.matchCount > model.theta_sub and self.accuracy > model.acc_sub:
            return True
        return False

    def isMoreGeneral(self,model, cl):
        if len(self.specifiedAttList) >= len(cl.specifiedAttList):
            return False
        for i in range(len(self.specifiedAttList)):
            attributeInfoType = model.env.formatData.attributeInfoType[self.specifiedAttList[i]]
            if self.specifiedAttList[i] not in cl.specifiedAttList:
                return False

            # Continuous
            if attributeInfoType:
                otherRef = cl.specifiedAttList.index(self.specifiedAttList[i])
                if self.condition[i][0] < cl.condition[otherRef][0]:
                    return False
                if self.condition[i][1] > cl.condition[otherRef][1]:
                    return False
        return True

