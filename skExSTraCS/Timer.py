import time


class Timer:
    def __init__(self):
        """ Initializes all Timer values for the algorithm """
        # Global Time objects
        self.globalStartRef = time.time()
        self.globalTime = 0.0
        self.addedTime = 0.0

        # Match Time Variables
        self.startRefMatching = 0.0
        self.globalMatching = 0.0

        # Covering Time Variables
        self.startRefCovering = 0.0
        self.globalCovering = 0.0

        # Deletion Time Variables
        self.startRefDeletion = 0.0
        self.globalDeletion = 0.0

        # Subsumption Time Variables
        self.startRefSubsumption = 0.0
        self.globalSubsumption = 0.0

        # Selection Time Variables
        self.startRefSelection = 0.0
        self.globalSelection = 0.0

        # Crossover Time Variables
        self.startRefCrossover = 0.0
        self.globalCrossover = 0.0

        # Mutation Time Variables
        self.startRefMutation = 0.0
        self.globalMutation = 0.0

        # Attribute Tracking and Feedback
        self.startRefAT = 0.0
        self.globalAT = 0.0

        # Expert Knowledge (EK)
        self.startRefEK = 0.0
        self.globalEK = 0.0

        # Initialization
        self.startRefInit = 0.0
        self.globalInit = 0.0

        # Add Classifier
        self.startRefAdd = 0.0
        self.globalAdd = 0.0

        # Evaluation Time Variables
        self.startRefEvaluation = 0.0
        self.globalEvaluation = 0.0

        # Rule Compaction
        self.startRefRuleCmp = 0.0
        self.globalRuleCmp = 0.0

    # ************************************************************
    def startTimeMatching(self):
        """ Tracks MatchSet Time """
        self.startRefMatching = time.time()

    def stopTimeMatching(self):
        """ Tracks MatchSet Time """
        diff = time.time() - self.startRefMatching
        self.globalMatching += diff

        # ************************************************************

    def startTimeCovering(self):
        """ Tracks MatchSet Time """
        self.startRefCovering = time.time()

    def stopTimeCovering(self):
        """ Tracks MatchSet Time """
        diff = time.time() - self.startRefCovering
        self.globalCovering += diff

        # ************************************************************

    def startTimeDeletion(self):
        """ Tracks Deletion Time """
        self.startRefDeletion = time.time()

    def stopTimeDeletion(self):
        """ Tracks Deletion Time """
        diff = time.time() - self.startRefDeletion
        self.globalDeletion += diff

    # ************************************************************
    def startTimeCrossover(self):
        """ Tracks Crossover Time """
        self.startRefCrossover = time.time()

    def stopTimeCrossover(self):
        """ Tracks Crossover Time """
        diff = time.time() - self.startRefCrossover
        self.globalCrossover += diff

    # ************************************************************
    def startTimeMutation(self):
        """ Tracks Mutation Time """
        self.startRefMutation = time.time()

    def stopTimeMutation(self):
        """ Tracks Mutation Time """
        diff = time.time() - self.startRefMutation
        self.globalMutation += diff

    # ************************************************************
    def startTimeSubsumption(self):
        """Tracks Subsumption Time """
        self.startRefSubsumption = time.time()

    def stopTimeSubsumption(self):
        """Tracks Subsumption Time """
        diff = time.time() - self.startRefSubsumption
        self.globalSubsumption += diff

        # ************************************************************

    def startTimeSelection(self):
        """ Tracks Selection Time """
        self.startRefSelection = time.time()

    def stopTimeSelection(self):
        """ Tracks Selection Time """
        diff = time.time() - self.startRefSelection
        self.globalSelection += diff

    # ************************************************************
    def startTimeEvaluation(self):
        """ Tracks Evaluation Time """
        self.startRefEvaluation = time.time()

    def stopTimeEvaluation(self):
        """ Tracks Evaluation Time """
        diff = time.time() - self.startRefEvaluation
        self.globalEvaluation += diff

        # ************************************************************

    def startTimeRuleCmp(self):
        """  """
        self.startRefRuleCmp = time.time()

    def stopTimeRuleCmp(self):
        """  """
        diff = time.time() - self.startRefRuleCmp
        self.globalRuleCmp += diff

    # ***********************************************************
    def startTimeAT(self):
        """  """
        self.startRefAT = time.time()

    def stopTimeAT(self):
        """  """
        diff = time.time() - self.startRefAT
        self.globalAT += diff

    # ***********************************************************
    def startTimeEK(self):
        """  """
        self.startRefEK = time.time()

    def stopTimeEK(self):
        """  """
        diff = time.time() - self.startRefEK
        self.globalEK += diff

    # ***********************************************************
    def startTimeInit(self):
        """  """
        self.startRefInit = time.time()

    def stopTimeInit(self):
        """  """
        diff = time.time() - self.startRefInit
        self.globalInit += diff

    # ***********************************************************
    def startTimeAdd(self):
        """  """
        self.startRefAdd = time.time()

    def stopTimeAdd(self):
        """  """
        diff = time.time() - self.startRefAdd
        self.globalAdd += diff

    # ***********************************************************

    def updateGlobalTimer(self):
        """ Update the global timer """
        self.globalTime = (time.time() - self.globalStartRef) + self.addedTime
