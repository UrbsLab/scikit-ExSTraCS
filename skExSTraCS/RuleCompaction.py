import copy
class RuleCompaction:
    def __init__(self,model):
        self.pop = model.population
        self.originalPopLength = len(model.population.popSet)
        self.originalTrainAcc = model.getFinalTrainingAccuracy(RC=True)

        if model.ruleCompaction == 'Fu1':
            self.approach_Fu1()
        elif model.ruleCompaction == 'Fu2':
            self.approach_Fu2()
        elif model.ruleCompaction == 'CRA2':
            self.approach_CRA2()
        elif model.ruleCompaction == 'QRC':
            self.approach_QRC()
        elif model.ruleCompaction == 'PDRC':
            self.approach_PDRC()
        elif model.ruleCompaction == 'QRF':
            self.approach_QRF()

        model.trackingObj.RCCount = self.originalPopLength - len(self.pop.popSet)

    def approach_Fu1(self,model):
        lastGood_popSet = sorted(self.pop.popSet, key=self.numerositySort)
        self.pop.popSet = lastGood_popSet[:]

        # STAGE 1----------------------------------------------------------------------------------------------------------------------
        keepGoing = True
        while keepGoing:
            del self.pop.popSet[0]  # Remove next classifier
            newAccuracy = model.getFinalTrainingAccuracy(RC=True)  # Perform classifier set training accuracy evaluation

            if newAccuracy < self.originalTrainAcc:
                keepGoing = False
                self.pop.popSet = lastGood_popSet[:]
            else:
                lastGood_popSet = self.pop.popSet[:]
            if len(self.pop.popSet) == 0:
                keepGoing = False

        # STAGE 2----------------------------------------------------------------------------------------------------------------------
        retainedClassifiers = []
        RefAccuracy = self.originalTrainAcc
        for i in range(len(self.pop.popSet)):
            heldClassifier = self.pop.popSet[0]
            del self.pop.popSet[0]
            newAccuracy = model.getFinalTrainingAccuracy(RC=True)  # Perform classifier set training accuracy evaluation

            if newAccuracy < RefAccuracy:
                retainedClassifiers.append(heldClassifier)
                RefAccuracy = newAccuracy

        self.pop.popSet = retainedClassifiers

        # STAGE 3----------------------------------------------------------------------------------------------------------------------
        finalClassifiers = []
        completelyGeneralRuleRef = None
        if len(self.pop.popSet) == 0:  # Stop Check
            keepGoing = False
        else:
            keepGoing = True

        matchCountList = [0.0 for v in range(len(self.pop.popSet))]
        for i in range(len(self.pop.popSet)):
            model.env.resetDataRef()
            for j in range(model.env.formatData.numTrainInstances):
                cl = self.pop.popSet[i]
                state = model.env.getTrainIsntance()[0]
                if cl.match(model,state):
                    matchCountList[i] += 1
                model.env.newInstance()

            if len(self.pop.popSet[i].condition) == 0:
                completelyGeneralRuleRef = i

        if completelyGeneralRuleRef != None:
            del matchCountList[completelyGeneralRuleRef]
            del self.pop.popSet[completelyGeneralRuleRef]

        tempEnv = copy.deepcopy(model.env)
        trainingData = tempEnv.formatData.trainFormatted
        while len(trainingData) > 0 and keepGoing:
            bestRef = None
            bestValue = None
            for i in range(len(matchCountList)):
                if bestValue == None or bestValue < matchCountList[i]:
                    bestRef = i
                    bestValue = matchCountList[i]

            if bestValue == 0.0 or len(self.pop.popSet) < 1:
                keepGoing = False
                continue

            # Update Training Data----------------------------------------------------------------------------------------------------
            matchedData = 0
            w = 0
            cl = self.pop.popSet[bestRef]
            for i in range(len(trainingData)):
                state = trainingData[w][0]
                doesMatch = cl.match(model,state)
                if doesMatch:
                    matchedData += 1
                    del trainingData[w]
                else:
                    w += 1
            if matchedData > 0:
                finalClassifiers.append(self.pop.popSet[bestRef])  # Add best classifier to final list - only do this if there are any remaining matching data instances for this rule!

            # Update classifier list
            del self.pop.popSet[bestRef]

            # re-calculate match count list
            matchCountList = [0.0 for v in range(len(self.pop.popSet))]
            for i in range(len(self.pop.popSet)):
                dataRef = 0
                for j in range(len(trainingData)):  # For each instance in training data
                    cl = self.pop.popSet[i]
                    state = trainingData[dataRef][0]
                    doesMatch = cl.match(model,state)
                    if doesMatch:
                        matchCountList[i] += 1
                    dataRef += 1

            if len(self.pop.popSet) == 0:
                keepGoing = False

        self.pop.popSet = finalClassifiers

    def approach_Fu2(self,model):
        lastGood_popSet = sorted(self.pop.popSet, key=self.numerositySort)
        self.pop.popSet = lastGood_popSet[:]

        # STAGE 1----------------------------------------------------------------------------------------------------------------------
        keepGoing = True
        while keepGoing:
            del self.pop.popSet[0]  # Remove next classifier
            newAccuracy = model.getFinalTrainingAccuracy(RC=True)  # Perform classifier set training accuracy evaluation
            if newAccuracy < self.originalTrainAcc:
                keepGoing = False
                self.pop.popSet = lastGood_popSet[:]
            else:
                lastGood_popSet = self.pop.popSet[:]
            if len(self.pop.popSet) == 0:
                keepGoing = False

        # STAGE 2----------------------------------------------------------------------------------------------------------------------
        retainedClassifiers = []
        RefAccuracy = self.originalTrainAcc
        for i in range(len(self.pop.popSet)):
            heldClassifier = self.pop.popSet[0]
            del self.pop.popSet[0]
            newAccuracy = model.getFinalTrainingAccuracy(RC=True)  # Perform classifier set training accuracy evaluation

            if newAccuracy < RefAccuracy:
                retainedClassifiers.append(heldClassifier)
                RefAccuracy = newAccuracy

        self.pop.popSet = retainedClassifiers

        # STAGE 3----------------------------------------------------------------------------------------------------------------------
        Sort_popSet = sorted(self.pop.popSet, key=self.numerositySort, reverse=True)
        self.pop.popSet = Sort_popSet[:]
        RefAccuracy = model.getFinalTrainingAccuracy(RC=True)

        for i in range(len(self.pop.popSet)):
            heldClassifier = self.pop.popSet[0]
            del self.pop.popSet[0]
            newAccuracy = model.getFinalTrainingAccuracy(RC=True)  # Perform classifier set training accuracy evaluation

            if newAccuracy < RefAccuracy:
                self.pop.popSet.append(heldClassifier)
            else:
                RefAccuracy = newAccuracy

    def approach_CRA2(self,model):
        retainedClassifiers = []
        matchSet = []
        correctSet = []
        model.env.resetDataRef()
        for j in range(model.env.formatData.numTrainInstances):
            state_phenotype = model.env.getTrainInstance()
            state = state_phenotype[0]
            phenotype = state_phenotype[1]

            # Create MatchSet
            for i in range(len(self.pop.popSet)):
                cl = self.pop.popSet[i]
                if cl.match(model,state):
                    matchSet.append(i)

            # Create CorrectSet
            for i in range(len(matchSet)):
                ref = matchSet[i]
                if self.pop.popSet[ref].phenotype == phenotype:
                    correctSet.append(ref)

            # Find the rule with highest accuracy, generality product
            highestValue = 0
            highestRef = 0
            for i in range(len(correctSet)):
                ref = correctSet[i]
                product = self.pop.popSet[ref].accuracy * (model.env.formatData.numAttributes - len(self.pop.popSet[ref].condition)) / float(model.env.formatData.numAttributes)
                if product > highestValue:
                    highestValue = product
                    highestRef = ref

            # If the rule is not already in the final ruleset, move it to the final ruleset
            if highestValue == 0 or self.pop.popSet[highestRef] in retainedClassifiers:
                pass
            else:
                retainedClassifiers.append(self.pop.popSet[highestRef])

            # Move to the next instance
            model.env.newInstance(True)
            matchSet = []
            correctSet = []

        self.pop.popSet = retainedClassifiers

    def approach_QRC(self,model):
        finalClassifiers = []
        if len(self.pop.popSet) == 0:  # Stop check
            keepGoing = False
        else:
            keepGoing = True

        lastGood_popSet = sorted(self.pop.popSet, key=self.accuracySort, reverse=True)
        self.pop.popSet = lastGood_popSet[:]

        tempEnv = copy.deepcopy(model.env)
        trainingData = tempEnv.formatData.trainFormatted

        while len(trainingData) > 0 and keepGoing:
            newTrainSet = []
            matchedData = 0
            for w in range(len(trainingData)):
                cl = self.pop.popSet[0]
                state = trainingData[w][0]
                doesMatch = cl.match(model,state)
                if doesMatch:
                    matchedData += 1
                else:
                    newTrainSet.append(trainingData[w])
            if matchedData > 0:
                finalClassifiers.append(self.pop.popSet[0])  # Add best classifier to final list - only do this if there are any remaining matching data instances for this rule!
            # Update classifier list and training set list
            trainingData = newTrainSet
            del self.pop.popSet[0]
            if len(self.pop.popSet) == 0:
                keepGoing = False

        self.pop.popSet = finalClassifiers

    def approach_PDRC(self,model):
        retainedClassifiers = []
        matchSet = []
        correctSet = []

        model.env.resetDataRef(True)
        for j in range(model.env.formatData.numTrainInstances):
            state_phenotype = model.env.getTrainInstance()
            state = state_phenotype[0]
            phenotype = state_phenotype[1]

            # Create Match Set
            for i in range(len(self.pop.popSet)):
                cl = self.pop.popSet[i]
                if cl.match(model,state):
                    matchSet.append(i)

            # Create Correct Set
            for i in range(len(matchSet)):
                ref = matchSet[i]
                if self.pop.popSet[ref].phenotype == phenotype:
                    correctSet.append(ref)

            # Find the rule with highest accuracy, generality and numerosity product
            highestValue = 0
            highestRef = 0
            for i in range(len(correctSet)):
                ref = correctSet[i]
                product = self.pop.popSet[ref].accuracy * (model.env.formatData.numAttributes - len(self.pop.popSet[ref].condition)) / float(model.env.formatData.numAttributes) * self.pop.popSet[ref].numerosity
                if product > highestValue:
                    highestValue = product
                    highestRef = ref

            # If the rule is not already in the final ruleset, move it to the final ruleset
            if highestValue == 0 or self.pop.popSet[highestRef] in retainedClassifiers:
                pass
            else:
                retainedClassifiers.append(self.pop.popSet[highestRef])

            # Move to the next instance
            model.env.newInstance(True)
            self.matchSet = []
            self.correctSet = []

            self.pop.popSet = retainedClassifiers

    def approach_QRF(self):
        retainedClassifiers = []
        for i in range(len(self.pop.popSet)):
            if self.pop.popSet[i].accuracy <= 0.5 or (
                    self.pop.popSet[i].correctCover == 1 and len(self.pop.popSet[i].specifiedAttList) > 1):
                pass
            else:
                retainedClassifiers.append(self.pop.popSet[i])
        self.pop.popSet = retainedClassifiers

    def accuracySort(self, cl):
        return cl.accuracy

    def numerositySort(self, cl):
        """ Sorts from smallest numerosity to largest """
        return cl.numerosity