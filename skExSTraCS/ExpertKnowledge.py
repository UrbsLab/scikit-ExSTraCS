import copy

class ExpertKnowledge:
    def __init__(self,model):
        self.scores = None

        if model.doExpertKnowledge:
            if not isinstance(model.expert_knowledge,list):
                self.scores = model.expert_knowledge.tolist()
            else:
                self.scores = model.expert_knowledge
        else:
            raise Exception("EK is invalid. This should never happen")

        self.adjustScores(model)

        ekRankFlip = sorted(range(len(self.scores)),key=self.scores.__getitem__)
        ekRankFlip.reverse()
        self.EKRank = ekRankFlip #List of best to worst scores by index
        '''
        tempEK = copy.deepcopy(self.scores)
        for i in range(len(self.scores)):
            bestEK = tempEK[0]
            bestC = 0
            for j in range(1,len(tempEK)):
                if tempEK[j] > bestEK:
                    bestEK = tempEK[j]
                    bestC = j
            self.EKRank.append(bestC)
            tempEK[bestC] = 0
        '''
        '''
        self.refList = []
        for i in range(len(self.scores)):
            self.refList.append(i)

        maxVal = max(self.scores)
        probList = []
        for i in range(model.env.formatData.numAttributes):
            if maxVal == 0.0:
                probList.append(0.5)
            else:
                probList.append(self.scores[i] / float(maxVal + maxVal * 0.01))
        self.EKprobabilityList = probList
        '''

        #ADDED: normalizes self.scores
        EKSum = sum(self.scores)
        for i in range(len(self.scores)):
            self.scores[i]/=EKSum

    def adjustScores(self,model):
        minEK = min(self.scores)
        if minEK <= 0: #Changed to <= 0 insteadd of <0
            for i in range(len(self.scores)):
                self.scores[i] = self.scores[i] - minEK + model.init_fitness #0.76225 training accuracy w/ init_fitness on 20B MP 5k iter vs 0.8022 accuracy w/o.
        if sum(self.scores) == 0:
            for i in range(len(self.scores)):
                self.scores[i] += 1