from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from skExSTraCS.Timer import Timer
from skExSTraCS.OfflineEnvironment import OfflineEnvironment

class ExSTraCS(BaseEstimator,ClassifierMixin):
    def __init__(self,learningIterations=100000,N=1000,nu=1,chi=0.8,upsilon=0.04,theta_GA=25,theta_del=20,theta_sub=20,
                 acc_sub=0.99,beta=0.2,delta=0.1,init_fitness=0.01,fitnessReduction=0.1,theta_sel=0.5,ruleSpecificityLimit=None,
                 doCorrectSetSubsumption=False,doGASubsumption=True,selectionMethod='tournament',doAttributeTracking=True,
                 doAttributeFeedback=True,useExpertKnowledge=True,expertKnowledgeSource=None,filterAlgorithm='multisurf',
                 turfPercent=0.05,reliefNeighbors=10,reliefSampleFraction=1,ruleCompaction='QRF',rebootFilename=None,
                 discreteAttributeLimit=10,specifiedAttributes=np.array([]),randomSeed=None):
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
        :param expertKnowledgeSource:       Must be String or None. If None, EK source is internal. If not, it is a file
        :param filterAlgorithm:             Must be String. relieff or surf or surfstar or multisurf
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
        self.expertKnowledgeSource = expertKnowledgeSource
        self.filterAlgorithm = filterAlgorithm
        self.turfPercent = turfPercent
        self.reliefNeighbors = reliefNeighbors
        self.reliefSampleFraction = reliefSampleFraction
        self.ruleCompaction = ruleCompaction
        self.rebootFilename = rebootFilename
        self.discreteAttributeLimit = discreteAttributeLimit
        self.specifiedAttributes = specifiedAttributes
        self.randomSeed = randomSeed

        self.hasTrained = False

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

            self.timer.stopTimeEK()
        if self.doAttributeTracking:
            self.timer.startTimeAT()

            self.timer.stopTimeAT()
        self.timer.stopTimeInit()

