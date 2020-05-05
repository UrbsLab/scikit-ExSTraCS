
from skExSTraCS.Classifier import Classifier

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
            cl.updateEpochStatus(model,model.iterationCount)

            if cl.match(state):
                self.matchSet.append(i)
                setNumerositySum += cl.numerosity

                if cl.phenotype == phenotype:
                    doCovering = False

        model.timer.stopTimeMatching()

        model.timer.startTimeCovering()
        while doCovering:
            newCl = Classifier(model)
            newCl.initializeByCovering(model,setNumerositySum,state,phenotype)
            self.addClassifierToPopulation(model,newCl,True)
            self.matchSet.append(len(self.popSet)-1)
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
            self.popSet[ref].updateFitness()

    def doCorrectSetSubsumption(self,model):
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