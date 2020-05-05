import copy

class ExpertKnowledge:
    def __init__(self,model):
        self.scores = None

        if model.expertKnowledge != None:
            self.scores = model.expertKnowledge.tolist()
        else:
            raise Exception("EK is invalid. This should never happen")

        self.adjustScores(model)

        self.EKSum = sum(self.scores)
        self.EKRank = []
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

    def adjustScores(self,model):
        minEK = min(self.scores)
        if minEK < 0:
            for i in range(len(self.scores)):
                self.scores[i] = self.scores[i] - minEK + model.init_Fitness