from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
import numpy as np
from skExSTraCS.Timer import Timer
from skExSTraCS.OfflineEnvironment import OfflineEnvironment
from skExSTraCS.ExpertKnowledge import ExpertKnowledge
from skExSTraCS.AttributeTracking import AttributeTracking
from skExSTraCS.ClassifierSet import ClassifierSet
from skExSTraCS.Prediction import Prediction
from skExSTraCS.RuleCompaction import RuleCompaction
from skExSTraCS.IterationRecord import IterationRecord
import copy
import time
import pickle
import random

class ExSTraCS(BaseEstimator,ClassifierMixin):
    def __init__(self,learningIterations=100000,N=1000,nu=1,chi=0.8,upsilon=0.04,theta_GA=25,theta_del=20,theta_sub=20,
                 acc_sub=0.99,beta=0.2,delta=0.1,init_fitness=0.01,fitnessReduction=0.1,theta_sel=0.5,ruleSpecificityLimit=None,
                 doCorrectSetSubsumption=False,doGASubsumption=True,selectionMethod='tournament',doAttributeTracking=True,
                 doAttributeFeedback=True,expertKnowledge=None,ruleCompaction='QRF',rebootFilename=None,
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
        :param expertKnowledge:             Must be np.ndarray or list or None. If None, don't use EK. If not, attribute filter scores are the array (in order)
        :param ruleCompaction:              Must be None or String. QRF or PDRC or QRC or CRA2 or Fu2 or Fu1. If None, no rule compaction is done
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
        #learningIterations
        if not self.checkIsInt(learningIterations):
            raise Exception("learningIterations param must be nonnegative integer")

        if learningIterations < 0:
            raise Exception("learningIterations param must be nonnegative integer")

        #N
        if not self.checkIsInt(N):
            raise Exception("N param must be nonnegative integer")

        if N < 0:
            raise Exception("N param must be nonnegative integer")

        #nu
        if not self.checkIsFloat(nu):
            raise Exception("nu param must be float")

        #chi
        if not self.checkIsFloat(chi):
            raise Exception("chi param must be float from 0 - 1")

        if chi < 0 or chi > 1:
            raise Exception("chi param must be float from 0 - 1")

        #upsilon
        if not self.checkIsFloat(upsilon):
            raise Exception("upsilon param must be float from 0 - 1")

        if upsilon < 0 or upsilon > 1:
            raise Exception("upsilon param must be float from 0 - 1")

        #theta_GA
        if not self.checkIsFloat(theta_GA):
            raise Exception("theta_GA param must be nonnegative float")

        if theta_GA < 0:
            raise Exception("theta_GA param must be nonnegative float")

        #theta_del
        if not self.checkIsInt(theta_del):
            raise Exception("theta_del param must be nonnegative integer")

        if theta_del < 0:
            raise Exception("theta_del param must be nonnegative integer")

        #theta_sub
        if not self.checkIsInt(theta_sub):
            raise Exception("theta_sub param must be nonnegative integer")

        if theta_sub < 0:
            raise Exception("theta_sub param must be nonnegative integer")

        #acc_sub
        if not self.checkIsFloat(acc_sub):
            raise Exception("acc_sub param must be float from 0 - 1")

        if acc_sub < 0 or acc_sub > 1:
            raise Exception("acc_sub param must be float from 0 - 1")

        #beta
        if not self.checkIsFloat(beta):
            raise Exception("beta param must be float")

        #delta
        if not self.checkIsFloat(delta):
            raise Exception("delta param must be float")

        #init_fitness
        if not self.checkIsFloat(init_fitness):
            raise Exception("init_fitness param must be float")

        #fitnessReduction
        if not self.checkIsFloat(fitnessReduction):
            raise Exception("fitnessReduction param must be float")

        #theta_sel
        if not self.checkIsFloat(theta_sel):
            raise Exception("theta_sel param must be float from 0 - 1")

        if theta_sel < 0 or theta_sel > 1:
            raise Exception("theta_sel param must be float from 0 - 1")

        #ruleSpecificityLimit
        if ruleSpecificityLimit != None:
            if not self.checkIsInt(ruleSpecificityLimit):
                raise Exception("ruleSpecificityLimit param must be nonnegative integer or None")

            if ruleSpecificityLimit < 0:
                raise Exception("ruleSpecificityLimit param must be nonnegative integer or None")

        #doCorrectSetSubsumption
        if not(isinstance(doCorrectSetSubsumption,bool)):
            raise Exception("doCorrectSetSubsumption param must be boolean")

        #doGASubsumption
        if not (isinstance(doGASubsumption, bool)):
            raise Exception("doGASubsumption param must be boolean")

        #selectionMethod
        if selectionMethod != "tournament" and selectionMethod != "roulette":
            raise Exception("selectionMethod param must be 'tournament' or 'roulette'")

        #doAttributeTracking
        if not(isinstance(doAttributeTracking,bool)):
            raise Exception("doAttributeTracking param must be boolean")

        #doAttributeFeedback
        if not(isinstance(doAttributeFeedback,bool)):
            raise Exception("doAttributeFeedback param must be boolean")

        #expertKnowledge
        if not (isinstance(expertKnowledge, np.ndarray)) and not (isinstance(expertKnowledge, list)) and expertKnowledge != None:
            raise Exception("expertKnowledge param must be None or list/ndarray")
        if isinstance(expertKnowledge,np.ndarray):
            expertKnowledge = expertKnowledge.tolist()

        #ruleCompaction
        if ruleCompaction != None and ruleCompaction != 'QRF' and ruleCompaction != 'PDRC' and ruleCompaction != 'QRC' and ruleCompaction != 'CRA2' and ruleCompaction != 'Fu2' and ruleCompaction != 'Fu1':
            raise Exception("ruleCompaction param must be None or 'QRF' or 'PDRC' or 'QRC' or 'CRA2' or 'Fu2' or 'Fu1'")

        #rebootFilename
        if rebootFilename != None and not isinstance(rebootFilename, str):
            raise Exception("rebootFilename param must be None or String from pickle")

        #discreteAttributeLimit
        if discreteAttributeLimit != "c" and discreteAttributeLimit != "d":
            try:
                dpl = int(discreteAttributeLimit)
                if not self.checkIsInt(discreteAttributeLimit):
                    raise Exception("discreteAttributeLimit param must be nonnegative integer or 'c' or 'd'")
                if dpl < 0:
                    raise Exception("discreteAttributeLimit param must be nonnegative integer or 'c' or 'd'")
            except:
                raise Exception("discreteAttributeLimit param must be nonnegative integer or 'c' or 'd'")

        #specifiedAttributes
        if not (isinstance(specifiedAttributes,np.ndarray)):
            raise Exception("specifiedAttributes param must be ndarray")

        for spAttr in specifiedAttributes:
            if not self.checkIsInt(spAttr):
                raise Exception("All specifiedAttributes elements param must be nonnegative integers")
            if int(spAttr) < 0:
                raise Exception("All specifiedAttributes elements param must be nonnegative integers")

        #trackAccuracyWhileFit
        if not(isinstance(trackAccuracyWhileFit,bool)):
            raise Exception("trackAccuracyWhileFit param must be boolean")

        #randomSeed
        if randomSeed != None:
            try:
                if not self.checkIsInt(randomSeed):
                    raise Exception("randomSeed param must be integer")
                random.seed(int(randomSeed))
                np.random.seed(int(randomSeed))
            except:
                raise Exception("randomSeed param must be integer or None")

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
        if not (isinstance(expertKnowledge, np.ndarray)) and not (isinstance(expertKnowledge, list)):
            self.doExpertKnowledge = False
        else:
            self.doExpertKnowledge = True
        self.expertKnowledge = expertKnowledge
        self.ruleCompaction = ruleCompaction
        self.rebootFilename = rebootFilename
        self.discreteAttributeLimit = discreteAttributeLimit
        self.specifiedAttributes = specifiedAttributes
        self.trackAccuracyWhileFit = trackAccuracyWhileFit
        self.randomSeed = randomSeed

        self.hasTrained = False
        self.trackingObj = tempTrackingObj()
        self.record = IterationRecord()

    def checkIsInt(self, num):
        try:
            n = float(num)
            if num - int(num) == 0:
                return True
            else:
                return False
        except:
            return False

    def checkIsFloat(self, num):
        try:
            n = float(num)
            return True
        except:
            return False

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

        self.iterationCount = 0
        self.trackingAccuracy = []
        self.movingAvgCount = 50
        self.env = OfflineEnvironment(X,y,self)
        aveGenerality = 0
        aveGeneralityFreq = min(self.env.formatData.numTrainInstances, int(self.learningIterations / 20) + 1)

        if self.rebootFilename == None:
            self.timer = Timer()
            self.population = ClassifierSet()
        else:
            self.rebootPopulation()

        self.timer.startTimeInit()

        if self.doExpertKnowledge:
            if len(self.expertKnowledge) != self.env.formatData.numAttributes:
                raise Exception("length of expertKnowledge param must match the # of data instance attributes")

        if self.doExpertKnowledge:
            self.timer.startTimeEK()
            self.EK = ExpertKnowledge(self)
            self.timer.stopTimeEK()
        if self.doAttributeTracking and (self.rebootFilename == None or (self.rebootFilename != None and self.AT == None)):
            self.timer.startTimeAT()
            self.AT = AttributeTracking(self)
            self.timer.stopTimeAT()
        elif not self.doAttributeTracking and self.rebootFilename == None:
            self.AT = None
        self.timer.stopTimeInit()

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

        self.preRCPop = copy.deepcopy(self.population.popSet)
        #Rule Compaction
        if self.ruleCompaction != None:
            self.trackingObj.resetAll()
            self.timer.startTimeRuleCmp()
            RuleCompaction(self)
            self.timer.stopTimeRuleCmp()
            self.timer.startTimeEvaluation()
            if len(self.trackingAccuracy) != 0:
                accuracy = sum(self.trackingAccuracy) / len(self.trackingAccuracy)
            else:
                accuracy = 0
            aveGenerality = self.population.getAveGenerality(self)
            self.trackingObj.macroPopSize = len(self.population.popSet)
            self.population.microPopSize = 0
            for rule in self.population.popSet:
                self.trackingObj.microPopSize += rule.numerosity
                self.population.microPopSize += rule.numerosity
            self.population.clearSets()
            self.trackingObj.avgIterAge = self.population.getInitStampAverage()
            self.timer.stopTimeEvaluation()
            self.timer.updateGlobalTimer()
            self.addToTracking(accuracy, aveGenerality)

        self.saveFinalMetrics()
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

    ##*************** Population Reboot ****************
    def saveFinalMetrics(self):
        self.finalMetrics = [self.learningIterations,self.timer.globalTime, self.timer.globalMatching,self.timer.globalCovering,
                             self.timer.globalCrossover, self.timer.globalMutation, self.timer.globalAT,self.timer.globalEK,
                             self.timer.globalInit, self.timer.globalAdd, self.timer.globalRuleCmp,self.timer.globalDeletion,
                             self.timer.globalSubsumption, self.timer.globalSelection, self.timer.globalEvaluation,copy.deepcopy(self.AT),
                             copy.deepcopy(self.population.popSet),self.preRCPop]

    def pickleModel(self,filename=None,saveRCPop=False):
        if self.hasTrained:
            if filename == None:
                filename = 'pickled'+str(int(time.time()))
            outfile = open(filename, 'wb')
            finalMetricsCopy = copy.deepcopy(self.finalMetrics)
            if saveRCPop==False:
                finalMetricsCopy.pop(len(self.finalMetrics) - 2)
            else:
                finalMetricsCopy.pop(len(self.finalMetrics) - 1)
            pickle.dump(finalMetricsCopy, outfile)
            outfile.close()

    def rebootPopulation(self):
        file = open(self.rebootFilename,'rb')
        rawData = pickle.load(file)
        file.close()

        popSet = rawData[len(rawData)-1]
        microPopSize = 0
        for rule in popSet:
            microPopSize += rule.numerosity
        set = ClassifierSet()
        set.popSet = popSet
        set.microPopSize = microPopSize
        self.population = set
        self.timer = Timer()
        self.timer.addedTime = rawData[1]
        self.timer.globalMatching = rawData[2]
        self.timer.globalCovering = rawData[3]
        self.timer.globalCrossover = rawData[4]
        self.timer.globalMutation = rawData[5]
        self.timer.globalAT = rawData[6]
        self.timer.globalEK = rawData[7]
        self.timer.globalInit = rawData[8]
        self.timer.globalAdd = rawData[9]
        self.timer.globalRuleCmp = rawData[10]
        self.timer.globalDeletion = rawData[11]
        self.timer.globalSubsumption = rawData[12]
        self.timer.globalSelection = rawData[13]
        self.timer.globalEvaluation = rawData[14]
        self.learningIterations += rawData[0]
        self.iterationCount += rawData[0]
        self.AT = rawData[15]

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
            probs = prediction.getProbabilities()
            predList.append(probs)
            self.population.clearSets()
        return np.array(predList)

    def score(self,X,y):
        predList = self.predict(X)
        return balanced_accuracy_score(y,predList)

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

    def getFinalAttributeTrackingSums(self):
        if self.hasTrained:
            return self.AT.getSumGlobalAttTrack(self)
        else:
            raise Exception("There are no final attribute tracking sums to return, as the ExSTraCS model has not been trained")

    def getFinalAttributeCooccurences(self,headers,maxNumAttributesToTrack=50):
        #Return a 2D list of [[attr1Name, attr2Name, specificity, accuracy weighted specificity]...]
        if self.hasTrained:
            if not isinstance(headers,np.ndarray):
                raise Exception("headers param must be ndarray")
            if len(headers.tolist()) != self.env.formatData.numAttributes:
                raise Exception("length of headers param must match the # of data instance attributes")

            if self.env.formatData.numAttributes <= maxNumAttributesToTrack:
                attList = []
                for i in range(0,self.env.formatData.numAttributes):
                    attList.append(i)
            else:
                attList = sorted(self.getFinalAttributeSpecificityList(),reverse=True)[:maxNumAttributesToTrack]

            comboList = []
            for i in range(0,len(attList)-1):
                for j in range(i+1,len(attList)):
                    comboList.append([headers[attList[i]],headers[attList[j]],0,0])

            for cl in self.population.popSet:
                counter = 0
                for i in range(0, len(attList) - 1):
                    for j in range(i + 1, len(attList)):
                        if attList[i] in cl.specifiedAttList and attList[j] in cl.specifiedAttList:
                            comboList[counter][2] += cl.numerosity
                            comboList[counter][3] += cl.numerosity * cl.accuracy
                        counter+=1

            return sorted(comboList,key=lambda test:test[3],reverse=True)

        else:
            raise Exception(
                "There are no final attribute cooccurences to return, as the ExSTraCS model has not been trained")

    ##Export Methods##
    def exportIterationTrackingData(self,filename='iterationData.csv'):
        if self.hasTrained:
            self.record.exportTrackingToCSV(filename)
        else:
            raise Exception("There is no tracking data to export, as the ExSTraCS model has not been trained")

    def exportFinalRulePopulation(self,headerNames=np.array([]),className="phenotype",filename='populationData.csv',DCAL=True,RCPopulation=False):
        if self.hasTrained:
            if RCPopulation:
                popSet = self.population.popSet
            else:
                popSet = self.preRCPop

            if DCAL:
                self.record.exportPopDCAL(self,popSet,headerNames,className,filename)
            else:
                self.record.exportPop(self,popSet,headerNames, className, filename)
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
