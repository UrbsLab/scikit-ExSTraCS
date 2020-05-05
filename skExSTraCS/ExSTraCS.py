from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from skExSTraCS.Timer import Timer
from skExSTraCS.OfflineEnvironment import OfflineEnvironment
from skExSTraCS.ExpertKnowledge import ExpertKnowledge
from skExSTraCS.AttributeTracking import AttributeTracking
from skExSTraCS.ClassifierSet import ClassifierSet
from skExSTraCS.Prediction import Prediction
from skExSTraCS.RuleCompaction import RuleCompaction
from skExSTraCS.IterationRecord import IterationRecord

class ExSTraCS(BaseEstimator,ClassifierMixin):
    def __init__(self,learningIterations=100000,N=1000,nu=1,chi=0.8,upsilon=0.04,theta_GA=25,theta_del=20,theta_sub=20,
                 acc_sub=0.99,beta=0.2,delta=0.1,init_fitness=0.01,fitnessReduction=0.1,theta_sel=0.5,ruleSpecificityLimit=None,
                 doCorrectSetSubsumption=False,doGASubsumption=True,selectionMethod='tournament',doAttributeTracking=True,
                 doAttributeFeedback=True,useExpertKnowledge=True,expertKnowledge=None,filterAlgorithm='multisurf',
                 turfPercent=0.05,reliefNeighbors=10,reliefSampleFraction=1,ruleCompaction='QRF',rebootFilename=None,
                 discreteAttributeLimit=10,specifiedAttributes=np.array([]),trackAccuracyWhileFit=False,randomSeed=None):
        '''
        :param learningIterations:          Must be nonnegative integer. The number of training cycles to run.
        :param N:                           Must be nonnegative integer. Maximum micro classifier population size (sum of classifier numerosities).
        :param nu:                          Must be a float. Power parameter (v) used to determine the importance of high accuracy when calculating fitness.
        :param chi:                         Must be float from 0 - 1. The probability of applying crossover in the GA (X).
        :param upsilon:                     Must be float from 0 - 1. The probability (u) of mutating an allele within an offspring
        :param theta_GA:                    Must be nonnegative float. The GA threshold. The GA is applied in a set when the average time (# of iterations) since the last GA in the correct set is greater than theta_GA.
        :param theta_del:                   Must be a nonnegative integer. The deletion experience threshold; The calculation of the deletion probability changes once this threshold is passed.
        :param theta_sub:                   Must be a nonnegative integer. The subsumption experience threshold
        :param acc_sub:                     Must be float from 0 - 1. Subsumption accuracy requirement
        :param beta:                        Must be float. Learning parameter; Used in calculating average correct set size
        :param delta:                       Must be float. Deletion parameter; Used in determining deletion vote calculation.
        :param init_fitness:                Must be float. The initial fitness for a new classifier. (typically very small, approaching but not equal to zero)
        :param fitnessReduction:            Must be float. Initial fitness reduction in GA offspring rules.
        :param theta_sel:                   Must be float from 0 - 1. The fraction of the correct set to be included in tournament selection.
        :param ruleSpecificityLimit:        Must be nonnegative integer or None. If not None, overrides the automatically computed RSL.
        :param doCorrectSetSubsumption:     Must be boolean. Determines if subsumption is done in [C] after [C] updates.
        :param doGASubsumption:             Must be boolean. Determines if subsumption is done between offspring and parents after GA
        :param selectionMethod:             Must be either "tournament" or "roulette". Determines GA selection method. Recommended: tournament
        :param doAttributeTracking:         Must be boolean. Whether to have AT
        :param doAttributeFeedback:         Must be boolean. Whether to have AF
        :param useExpertKnowledge:          Must be boolean. Whether to use EK
        :param expertKnowledge:             Must be np.ndarray or None. If None, EK source is internal. If not, attribute filter scores are the array (in order)
        :param filterAlgorithm:             Must be String. relieff or surf or surfstar or multisurf or relieff_turf or surf_turf or surfstar_turf or multisurf_turf
        :param turfPercent:                 Must be float from 0 - 1.
        :param reliefNeighbors:             Must be nonnegative integer. The # of neighbors considered in Relief calculations
        :param reliefSampleFraction:        Must be float from 0 - 1. The # of EK weight algorithm iterations
        :param ruleCompaction:              Must be None or String. QRT or PDRC or QRC or CRA2 or Fu2 or Fu1. If None, no rule compaction is done
        :param rebootFilename:              Must be String or None. Filename of pickled model to be rebooted
        :param discreteAttributeLimit:      Must be nonnegative integer OR "c" OR "d". Multipurpose param. If it is a nonnegative integer, discreteAttributeLimit determines the threshold that determines
                                            if an attribute will be treated as a continuous or discrete attribute. For example, if discreteAttributeLimit == 10, if an attribute has more than 10 unique
                                            values in the dataset, the attribute will be continuous. If the attribute has 10 or less unique values, it will be discrete. Alternatively,
                                            discreteAttributeLimit can take the value of "c" or "d". See next param for this.
        :param specifiedAttributes:         Must be an ndarray type of nonnegative integer attributeIndices (zero indexed).
                                            If "c", attributes specified by index in this param will be continuous and the rest will be discrete. If "d", attributes specified by index in this
                                            param will be discrete and the rest will be continuous.
                                            If this value is given, and discreteAttributeLimit is not "c" or "d", discreteAttributeLimit overrides this specification
        :param trackAccuracyWhileFit        Must be boolean. Determines if accuracy is tracked during model training
        :param randomSeed:                  Must be an integer or None. Set a constant random seed value to some integer
        '''

        self.learningIterations = learningIterations
        self.N = N
        self.nu = nu
        self.chi = chi
        self.upsilon = upsilon
        self.theta_GA = theta_GA
        self.theta_del = theta_del
        self.theta_sub = theta_sub
        self.acc_sub = acc_sub
        self.beta = beta
        self.delta = delta
        self.init_fitness = init_fitness
        self.fitnessReduction = fitnessReduction
        self.theta_sel = theta_sel
        self.ruleSpecificityLimit = ruleSpecificityLimit
        self.doCorrectSetSubsumption = doCorrectSetSubsumption
        self.doGASubsumption = doGASubsumption
        self.selectionMethod = selectionMethod
        self.doAttributeTracking = doAttributeTracking
        self.doAttributeFeedback = doAttributeFeedback
        self.useExpertKnowledge = useExpertKnowledge
        self.expertKnowledge = expertKnowledge
        self.filterAlgorithm = filterAlgorithm
        self.turfPercent = turfPercent
        self.reliefNeighbors = reliefNeighbors
        self.reliefSampleFraction = reliefSampleFraction
        self.ruleCompaction = ruleCompaction
        self.rebootFilename = rebootFilename
        self.discreteAttributeLimit = discreteAttributeLimit
        self.specifiedAttributes = specifiedAttributes
        self.trackAccuracyWhileFit = trackAccuracyWhileFit
        self.randomSeed = randomSeed

        self.hasTrained = False
        self.trackingObj = tempTrackingObj()
        self.record = IterationRecord()

    ##*************** Fit ****************
    def fit(self, X, y):
        """Scikit-learn required: Supervised training of exstracs
             Parameters
            X: array-like {n_samples, n_features} Training instances. ALL INSTANCE ATTRIBUTES MUST BE NUMERIC or NAN
            y: array-like {n_samples} Training labels. ALL INSTANCE PHENOTYPES MUST BE NUMERIC NOT NAN OR OTHER TYPE
            Returns self
        """
        # If trained already, raise Exception
        if self.hasTrained:
            raise Exception("Cannot train already trained model again")

        # Check if X and Y are numeric
        try:
            for instance in X:
                for value in instance:
                    if not (np.isnan(value)):
                        float(value)
            for value in y:
                float(value)
        except:
            raise Exception("X and y must be fully numeric")

        self.timer = Timer()
        self.timer.startTimeInit()
        self.env = OfflineEnvironment(X,y,self)

        if self.useExpertKnowledge:
            self.timer.startTimeEK()
            self.EK = ExpertKnowledge(self)
            self.timer.stopTimeEK()
        if self.doAttributeTracking:
            self.timer.startTimeAT()
            self.AT = AttributeTracking(self)
            self.timer.stopTimeAT()
        self.timer.stopTimeInit()

        self.population = ClassifierSet()
        self.iterationCount = 0

        self.trackingAccuracy = []
        self.movingAvgCount = 50
        aveGenerality = 0
        aveGeneralityFreq = min(self.env.formatData.numTrainInstances, int(self.learningIterations / 20) + 1)

        while self.iterationCount < self.learningIterations:
            state_phenotype = self.env.getTrainInstance()
            self.runIteration(state_phenotype)

            self.timer.startTimeEvaluation()
            if self.iterationCount % aveGeneralityFreq == aveGeneralityFreq - 1:
                aveGenerality = self.population.getAveGenerality(self)

            if len(self.trackingAccuracy) != 0:
                accuracy = sum(self.trackingAccuracy)/len(self.trackingAccuracy)
            else:
                accuracy = 0

            self.timer.updateGlobalTimer()
            self.addToTracking(accuracy,aveGenerality)
            self.timer.stopTimeEvaluation()

            # Increment Instance & Iteration
            self.iterationCount += 1
            self.env.newInstance()

        #Rule Compaction
        if self.ruleCompaction != None:
            self.trackingObj.resetAll()
            self.timer.startTimeRuleCmp()
            RuleCompaction(self)
            self.timer.stopTimeRuleCmp()
            self.addToTracking(accuracy, aveGenerality)

        self.hasTrained = True
        return self

    def addToTracking(self,accuracy,aveGenerality):
        self.record.addToTracking(self.iterationCount, accuracy, aveGenerality, self.trackingObj.macroPopSize,
                                  self.trackingObj.microPopSize, self.trackingObj.matchSetSize,self.trackingObj.correctSetSize,
                                  self.trackingObj.avgIterAge, self.trackingObj.subsumptionCount,self.trackingObj.crossOverCount,
                                  self.trackingObj.mutationCount, self.trackingObj.coveringCount,self.trackingObj.deletionCount,
                                  self.trackingObj.RCCount, self.timer.globalTime, self.timer.globalMatching,self.timer.globalCovering,
                                  self.timer.globalCrossover, self.timer.globalMutation, self.timer.globalAT,self.timer.globalEK,
                                  self.timer.globalInit, self.timer.globalAdd, self.timer.globalRuleCmp,self.timer.globalDeletion,
                                  self.timer.globalSubsumption, self.timer.globalSelection, self.timer.globalEvaluation)

    def runIteration(self,state_phenotype):
        # Reset tracking object counters
        self.trackingObj.resetAll()

        #Make [M]
        self.population.makeMatchSet(self,state_phenotype)

        #Track Training Accuracy
        if self.trackAccuracyWhileFit:
            self.timer.startTimeEvaluation()
            prediction = Prediction(self,self.population)
            phenotypePrediction = prediction.getDecision()

            if phenotypePrediction == state_phenotype[1]:
                if len(self.trackingAccuracy) == self.movingAvgCount:
                    del self.trackingAccuracy[0]
                self.trackingAccuracy.append(1)
            else:
                if len(self.trackingAccuracy) == self.movingAvgCount:
                    del self.trackingAccuracy[0]
                self.trackingAccuracy.append(0)

            self.timer.stopTimeEvaluation()

        #Make [C]
        self.population.makeCorrectSet(state_phenotype[1])

        #Update Parameters
        self.population.updateSets(self)

        #[C] Subsumption
        if self.doCorrectSetSubsumption:
            self.timer.startTimeSubsumption()
            self.population.doCorrectSetSubsumption(self)
            self.timer.stopTimeSubsumption()

        #AT
        if self.doAttributeTracking:
            self.timer.startTimeAT()
            self.AT.updateAttTrack(self,self.population)
            if self.doAttributeFeedback:
                self.AT.updatePercent(self)
                self.AT.genTrackProb(self)
            self.timer.stopTimeAT()

        #Run GA
        self.population.runGA(self,state_phenotype[0],state_phenotype[1])

        #Deletion
        self.population.deletion(self)

        self.trackingObj.macroPopSize = len(self.population.popSet)
        self.trackingObj.microPopSize = self.population.microPopSize
        self.trackingObj.matchSetSize = len(self.population.matchSet)
        self.trackingObj.correctSetSize = len(self.population.correctSet)
        self.trackingObj.avgIterAge = self.population.getInitStampAverage()

        #Clear Sets
        self.population.clearSets()

    ##*************** Predict and Score ****************
    def predict(self, X):
        """Scikit-learn required: Test Accuracy of ExSTraCS
            Parameters
            X: array-like {n_samples, n_features} Test instances to classify. ALL INSTANCE ATTRIBUTES MUST BE NUMERIC

            Returns
            y: array-like {n_samples} Classifications.
        """
        try:
            for instance in X:
                for value in instance:
                    if not (np.isnan(value)):
                        float(value)
        except:
            raise Exception("X must be fully numeric")

        instances = X.shape[0]
        predList = []

        for inst in range(instances):
            state = X[inst]
            self.population.makeEvalMatchSet(self,state)
            prediction = Prediction(self, self.population)
            phenotypeSelection = prediction.getDecision()
            predList.append(phenotypeSelection)
            self.population.clearSets()
        return np.array(predList)

    def predict_proba(self, X):
        """Scikit-learn required: Test Accuracy of eLCS
            Parameters
            X: array-like {n_samples, n_features} Test instances to classify. ALL INSTANCE ATTRIBUTES MUST BE NUMERIC

            Returns
            y: array-like {n_samples} Classifications.
        """
        try:
            for instance in X:
                for value in instance:
                    if not (np.isnan(value)):
                        float(value)
        except:
            raise Exception("X must be fully numeric")

        instances = X.shape[0]
        predList = []

        for inst in range(instances):
            state = X[inst]
            self.population.makeEvalMatchSet(self,state)
            prediction = Prediction(self, self.population)
            probs = prediction.getProbabilities()
            predList.append(probs)
            self.population.clearSets()
        return np.array(predList)

    def score(self,X,y):
        predList = self.predict(X)
        return balanced_accuracy_score(y, predList) #Make it balanced accuracy

    ##*************** More Evaluation Methods ****************

    def getFinalTrainingAccuracy(self,RC=False):
        if self.hasTrained or RC:
            originalTrainingData = self.env.formatData.savedRawTrainingData
            return self.score(originalTrainingData[0], originalTrainingData[1])
        else:
            raise Exception("There is no final training accuracy to return, as the ExSTraCS model has not been trained")

    def getFinalInstanceCoverage(self):
        if self.hasTrained:
            numCovered = 0
            originalTrainingData = self.env.formatData.savedRawTrainingData
            for state in originalTrainingData[0]:
                self.population.makeEvalMatchSet(self,state)
                predictionArray = Prediction(self, self.population)
                if predictionArray.hasMatch:
                    numCovered += 1
                self.population.clearSets()
            return numCovered/len(originalTrainingData[0])
        else:
            raise Exception("There is no final instance coverage to return, as the ExSTraCS model has not been trained")

    def getFinalAttributeSpecificityList(self):
        if self.hasTrained:
            return self.population.getAttributeSpecificityList(self)
        else:
            raise Exception("There is no final attribute specificity list to return, as the ExSTraCS model has not been trained")

    def getFinalAttributeAccuracyList(self):
        if self.hasTrained:
            return self.population.getAttributeAccuracyList(self)
        else:
            raise Exception("There is no final attribute accuracy list to return, as the ExSTraCS model has not been trained")

    ##Export Methods##
    def exportIterationTrackingData(self,filename='iterationData.csv'):
        if self.hasTrained:
            self.record.exportTrackingToCSV(filename)
        else:
            raise Exception("There is no tracking data to export, as the eLCS model has not been trained")

    def exportFinalRulePopulation(self,headerNames=np.array([]),className="phenotype",filename='populationData.csv',DCAL=True):
        if self.hasTrained:
            if DCAL:
                self.record.exportPopDCAL(self,headerNames,className,filename)
            else:
                self.record.exportPop(self, headerNames, className, filename)
        else:
            raise Exception("There is no rule population to export, as the ExSTraCS model has not been trained")

class tempTrackingObj():
    #Tracks stats of every iteration (except accuracy, avg generality, and times)
    def __init__(self):
        self.macroPopSize = 0
        self.microPopSize = 0
        self.matchSetSize = 0
        self.correctSetSize = 0
        self.avgIterAge = 0
        self.subsumptionCount = 0
        self.crossOverCount = 0
        self.mutationCount = 0
        self.coveringCount = 0
        self.deletionCount = 0
        self.RCCount = 0

    def resetAll(self):
        self.macroPopSize = 0
        self.microPopSize = 0
        self.matchSetSize = 0
        self.correctSetSize = 0
        self.avgIterAge = 0
        self.subsumptionCount = 0
        self.crossOverCount = 0
        self.mutationCount = 0
        self.coveringCount = 0
        self.deletionCount = 0
        self.RCCount = 0
