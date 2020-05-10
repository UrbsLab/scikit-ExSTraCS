
from skExSTraCS.Classifier import Classifier
import copy
import random

class ClassifierSet:
    def __init__(self):
        self.popSet = []  # List of classifiers/rules
        self.matchSet = []  # List of references to rules in population that match
        self.correctSet = []  # List of references to rules in population that both match and specify correct phenotype
        self.microPopSize = 0

    def makeMatchSet(self,model,state_phenotype):
        state = state_phenotype[0]
        phenotype = state_phenotype[1]
        doCovering = True
        setNumerositySum = 0

        model.timer.startTimeMatching()
        for i in range(len(self.popSet)):
            cl = self.popSet[i]
            cl.updateEpochStatus(model)

            if cl.match(model,state):
                self.matchSet.append(i)
                setNumerositySum += cl.numerosity

                if cl.phenotype == phenotype:
                    doCovering = False

        model.timer.stopTimeMatching()

        model.timer.startTimeCovering()
        while doCovering:
            newCl = Classifier(model)
            newCl.initializeByCovering(model,setNumerositySum+1,state,phenotype)
            if len(newCl.specifiedAttList) > 0: #ADDED CHECK TO PREVENT FULLY GENERALIZED RULES
                self.addClassifierToPopulation(model,newCl,True)
                self.matchSet.append(len(self.popSet)-1)
                model.trackingObj.coveringCount += 1
                doCovering = False
        model.timer.stopTimeCovering()

    def addClassifierToPopulation(self,model,cl,covering):
        model.timer.startTimeAdd()
        oldCl = None
        if not covering:
            oldCl = self.getIdenticalClassifier(cl)
        if oldCl != None:
            oldCl.updateNumerosity(1)
            self.microPopSize += 1
        else:
            self.popSet.append(cl)
            self.microPopSize += 1
        model.timer.stopTimeAdd()

    def getIdenticalClassifier(self,newCl):
        for cl in self.popSet:
            if newCl.equals(cl):
                return cl
        return None

    def makeCorrectSet(self,phenotype):
        for i in range(len(self.matchSet)):
            ref = self.matchSet[i]
            if self.popSet[ref].phenotype == phenotype:
                self.correctSet.append(ref)

    def updateSets(self,model):
        """ Updates all relevant parameters in the current match and correct sets. """
        matchSetNumerosity = 0
        for ref in self.matchSet:
            matchSetNumerosity += self.popSet[ref].numerosity

        for ref in self.matchSet:
            self.popSet[ref].updateExperience()
            self.popSet[ref].updateMatchSetSize(model,matchSetNumerosity)  # Moved to match set to be like GHCS
            if ref in self.correctSet:
                self.popSet[ref].updateCorrect()

            self.popSet[ref].updateAccuracy()
            self.popSet[ref].updateFitness(model)

    def do_correct_set_subsumption(self,model):
        subsumer = None
        for ref in self.correctSet:
            cl = self.popSet[ref]
            if cl.isSubsumer(model):
                if subsumer == None or cl.isMoreGeneral(model,subsumer):
                    subsumer = cl

        if subsumer != None:
            i = 0
            while i < len(self.correctSet):
                ref = self.correctSet[i]
                if subsumer.isMoreGeneral(model,self.popSet[ref]):
                    model.trackingObj.subsumptionCount += 1
                    model.trackingObj.subsumptionCount += 1
                    subsumer.updateNumerosity(self.popSet[ref].numerosity)
                    self.removeMacroClassifier(ref)
                    self.deleteFromMatchSet(ref)
                    self.deleteFromCorrectSet(ref)
                    i -= 1
                i+=1

    def removeMacroClassifier(self, ref):
        """ Removes the specified (macro-) classifier from the population. """
        self.popSet.pop(ref)

    def deleteFromMatchSet(self, deleteRef):
        """ Delete reference to classifier in population, contained in self.matchSet."""
        if deleteRef in self.matchSet:
            self.matchSet.remove(deleteRef)
        # Update match set reference list--------
        for j in range(len(self.matchSet)):
            ref = self.matchSet[j]
            if ref > deleteRef:
                self.matchSet[j] -= 1

    def deleteFromCorrectSet(self, deleteRef):
        """ Delete reference to classifier in population, contained in self.matchSet."""
        if deleteRef in self.correctSet:
            self.correctSet.remove(deleteRef)
        # Update match set reference list--------
        for j in range(len(self.correctSet)):
            ref = self.correctSet[j]
            if ref > deleteRef:
                self.correctSet[j] -= 1

    def runGA(self,model,state,phenotype):
        if model.iterationCount - self.getIterStampAverage() < model.theta_GA:
            return
        self.setIterStamps(model.iterationCount)

        changed = False

        #Select Parents
        model.timer.startTimeSelection()
        if model.selection_method == "roulette":
            selectList = self.selectClassifierRW()
            clP1 = selectList[0]
            clP2 = selectList[1]
        elif model.selection_method == "tournament":
            selectList = self.selectClassifierT(model)
            clP1 = selectList[0]
            clP2 = selectList[1]
        model.timer.stopTimeSelection()

        #Create Offspring Copies
        cl1 = Classifier(model)
        cl1.initializeByCopy(clP1,model.iterationCount)
        cl2 = Classifier(model)
        if clP2 == None:
            cl2.initializeByCopy(clP1,model.iterationCount)
        else:
            cl2.initializeByCopy(clP2,model.iterationCount)

        #Crossover
        if not cl1.equals(cl2) and random.random() < model.chi:
            model.timer.startTimeCrossover()
            changed = cl1.uniformCrossover(model,cl2)
            model.timer.stopTimeCrossover()

        if changed:
            cl1.setAccuracy((cl1.accuracy + cl2.accuracy)/2.0)
            cl1.setFitness(model.fitness_reduction * (cl1.fitness + cl2.fitness)/2.0)
            cl2.setAccuracy(cl1.accuracy)
            cl2.setFitness(cl1.fitness)
        else:
            cl1.setFitness(model.fitness_reduction * cl1.fitness)
            cl2.setFitness(model.fitness_reduction * cl2.fitness)

        #Mutation
        model.timer.startTimeMutation()
        nowchanged = cl1.mutation(model,state)
        howaboutnow = cl2.mutation(model,state)
        model.timer.stopTimeMutation()

        if model.env.formatData.continuousCount > 0:
            cl1.rangeCheck(model)
            cl2.rangeCheck(model)

        if changed or nowchanged or howaboutnow:
            if nowchanged:
                model.trackingObj.mutationCount += 1
            if howaboutnow:
                model.trackingObj.mutationCount += 1
            if changed:
                model.trackingObj.crossOverCount += 1
            self.insertDiscoveredClassifiers(model,cl1, cl2, clP1, clP2) #Includes subsumption if activated.

    def insertDiscoveredClassifiers(self,model,cl1,cl2,clP1,clP2):
        if model.do_GA_subsumption:
            model.timer.startTimeSubsumption()
            if len(cl1.specifiedAttList) > 0:
                self.subsumeClassifier(model,cl1, clP1, clP2)
            if len(cl2.specifiedAttList) > 0:
                self.subsumeClassifier(model,cl2, clP1, clP2)
            model.timer.stopTimeSubsumption()
        else:
            if len(cl1.specifiedAttList) > 0:
                self.addClassifierToPopulation(model,cl1,False)
            if len(cl2.specifiedAttList) > 0:
                self.addClassifierToPopulation(model,cl2,False)

    def subsumeClassifier(self, model,cl, cl1P, cl2P):
        """ Tries to subsume a classifier in the parents. If no subsumption is possible it tries to subsume it in the current set. """
        if cl1P!=None and cl1P.subsumes(model,cl):
            self.microPopSize += 1
            cl1P.updateNumerosity(1)
            model.trackingObj.subsumptionCount+=1
        elif cl2P!=None and cl2P.subsumes(model,cl):
            self.microPopSize += 1
            cl2P.updateNumerosity(1)
            model.trackingObj.subsumptionCount += 1
        else:
            if len(cl.specifiedAttList) > 0:
                self.addClassifierToPopulation(model, cl, False)

    def selectClassifierRW(self):
        setList = copy.deepcopy(self.correctSet)

        if len(setList) > 2:
            selectList = [None,None]
            currentCount = 0

            while currentCount < 2:
                fitSum = self.getFitnessSum(setList)

                choiceP = random.random() * fitSum
                i = 0
                sumCl = self.popSet[setList[i]].fitness
                while choiceP > sumCl:
                    i = i + 1
                    sumCl += self.popSet[setList[i]].fitness

                selectList[currentCount] = self.popSet[setList[i]]
                setList.remove(setList[i])
                currentCount += 1

        elif len(setList) == 2:
            selectList = [self.popSet[setList[0]], self.popSet[setList[1]]]
        elif len(setList) == 1:
            selectList = [self.popSet[setList[0]], self.popSet[setList[0]]]

        return selectList

    def getFitnessSum(self, setList):
        """ Returns the sum of the fitnesses of all classifiers in the set. """
        sumCl = 0.0
        for i in range(len(setList)):
            ref = setList[i]
            sumCl += self.popSet[ref].fitness
        return sumCl

    def selectClassifierT(self,model):
        selectList = [None, None]
        currentCount = 0
        setList = self.correctSet

        while currentCount < 2:
            tSize = int(len(setList) * model.theta_sel)

            #Select tSize elements from correctSet
            posList = random.sample(setList,tSize)

            bestF = 0
            bestC = self.correctSet[0]
            for j in posList:
                if self.popSet[j].fitness > bestF:
                    bestF = self.popSet[j].fitness
                    bestC = j

            selectList[currentCount] = self.popSet[bestC]
            currentCount += 1

        return selectList

    def getIterStampAverage(self):
        """ Returns the average of the time stamps in the correct set. """
        sumCl=0
        numSum=0
        for i in range(len(self.correctSet)):
            ref = self.correctSet[i]
            sumCl += self.popSet[ref].timeStampGA * self.popSet[ref].numerosity
            numSum += self.popSet[ref].numerosity #numerosity sum of correct set
        if numSum != 0:
            return sumCl / float(numSum)
        else:
            return 0

    def getInitStampAverage(self):
        sumCl = 0.0
        numSum = 0.0
        for i in range(len(self.correctSet)):
            ref = self.correctSet[i]
            sumCl += self.popSet[ref].initTimeStamp * self.popSet[ref].numerosity
            numSum += self.popSet[ref].numerosity
        if numSum != 0:
            return sumCl / float(numSum)
        else:
            return 0

    def setIterStamps(self, iterationCount):
        """ Sets the time stamp of all classifiers in the set to the current time. The current time
        is the number of exploration steps executed so far.  """
        for i in range(len(self.correctSet)):
            ref = self.correctSet[i]
            self.popSet[ref].updateTimeStamp(iterationCount)

    def deletion(self,model):
        model.timer.startTimeDeletion()
        while self.microPopSize > model.N:
            self.deleteFromPopulation(model)
        model.timer.stopTimeDeletion()

    def deleteFromPopulation(self,model):
        meanFitness = self.getPopFitnessSum() / float(self.microPopSize)

        sumCl = 0.0
        voteList = []
        for cl in self.popSet:
            vote = cl.getDelProp(model,meanFitness)
            sumCl += vote
            voteList.append(vote)

        i = 0
        for cl in self.popSet:
            cl.deletionProb = voteList[i] / sumCl
            i += 1

        choicePoint = sumCl * random.random()  # Determine the choice point

        newSum = 0.0
        for i in range(len(voteList)):
            cl = self.popSet[i]
            newSum = newSum + voteList[i]
            if newSum > choicePoint:  # Select classifier for deletion
                # Delete classifier----------------------------------
                cl.updateNumerosity(-1)
                self.microPopSize -= 1
                if cl.numerosity < 1:  # When all micro-classifiers for a given classifier have been depleted.
                    self.removeMacroClassifier(i)
                    self.deleteFromMatchSet(i)
                    self.deleteFromCorrectSet(i)
                    model.trackingObj.deletionCount += 1
                return

    def getPopFitnessSum(self):
        """ Returns the sum of the fitnesses of all classifiers in the set. """
        sumCl=0.0
        for cl in self.popSet:
            sumCl += cl.fitness *cl.numerosity
        return sumCl

    def clearSets(self):
        """ Clears out references in the match and correct sets for the next learning iteration. """
        self.matchSet = []
        self.correctSet = []

    def getAveGenerality(self,model):
        genSum = 0
        for cl in self.popSet:
            genSum += ((model.env.formatData.numAttributes - len(cl.condition))/float(model.env.formatData.numAttributes))*cl.numerosity
        if self.microPopSize == 0:
            aveGenerality = 0
        else:
            aveGenerality = genSum/float(self.microPopSize)
        return aveGenerality

    def makeEvalMatchSet(self,model,state):
        for i in range(len(self.popSet)):
            cl = self.popSet[i]
            if cl.match(model,state):
                self.matchSet.append(i)

    def getAttributeSpecificityList(self,model):
        attributeSpecList = []
        for i in range(model.env.formatData.numAttributes):
            attributeSpecList.append(0)
        for cl in self.popSet:
            for ref in cl.specifiedAttList:
                attributeSpecList[ref] += cl.numerosity
        return attributeSpecList

    def getAttributeAccuracyList(self,model):
        attributeAccList = []
        for i in range(model.env.formatData.numAttributes):
            attributeAccList.append(0.0)
        for cl in self.popSet:
            for ref in cl.specifiedAttList:
                attributeAccList[ref] += cl.numerosity * cl.accuracy
        return attributeAccList