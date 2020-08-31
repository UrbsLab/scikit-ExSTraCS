import random
import copy
import numpy as np

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

    def initializeByCopy(self,toCopy,iterationCount):
        self.specifiedAttList = copy.deepcopy(toCopy.specifiedAttList)
        self.condition = copy.deepcopy(toCopy.condition)
        self.phenotype = copy.deepcopy(toCopy.phenotype)
        self.timeStampGA = iterationCount
        self.initTimeStamp = iterationCount
        self.aveMatchSetSize = copy.deepcopy(toCopy.aveMatchSetSize)
        self.fitness = toCopy.fitness
        self.accuracy = toCopy.accuracy

    def initializeByCovering(self,model,setSize,state,phenotype):
        self.timeStampGA = model.iterationCount
        self.initTimeStamp = model.iterationCount
        self.aveMatchSetSize = setSize
        self.phenotype = phenotype

        toSpecify = random.randint(1, model.rule_specificity_limit)
        if model.doExpertKnowledge:
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
                if state[attRef] != None:
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

    def subsumes(self,model,cl):
        return cl.phenotype == self.phenotype and self.isSubsumer(model) and self.isMoreGeneral(model,cl)

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

    def updateTimeStamp(self, ts):
        """ Sets the time stamp of the classifier. """
        self.timeStampGA = ts

    def uniformCrossover(self,model,cl):
        p_self_specifiedAttList = copy.deepcopy(self.specifiedAttList)
        p_cl_specifiedAttList = copy.deepcopy(cl.specifiedAttList)

        useAT = model.do_attribute_feedback and random.random() < model.AT.percent

        comboAttList = []
        for i in p_self_specifiedAttList:
            comboAttList.append(i)
        for i in p_cl_specifiedAttList:
            if i not in comboAttList:
                comboAttList.append(i)
            elif not model.env.formatData.attributeInfoType[i]:  # Attribute specified in both parents, and the attribute is discrete (then no reason to cross over)
                comboAttList.remove(i)
        comboAttList.sort()

        changed = False
        for attRef in comboAttList:
            attributeInfoType = model.env.formatData.attributeInfoType[attRef]
            if useAT:
                probability = model.AT.getTrackProb()[attRef]
            else:
                probability = 0.5

            ref = 0
            if attRef in p_self_specifiedAttList:
                ref += 1
            if attRef in p_cl_specifiedAttList:
                ref += 1

            if ref == 0:
                pass
            elif ref == 1:
                if attRef in p_self_specifiedAttList and random.random() > probability:
                    i = self.specifiedAttList.index(attRef)
                    cl.condition.append(self.condition.pop(i))

                    cl.specifiedAttList.append(attRef)
                    self.specifiedAttList.remove(attRef)
                    changed = True

                if attRef in p_cl_specifiedAttList and random.random() < probability:
                    i = cl.specifiedAttList.index(attRef)
                    self.condition.append(cl.condition.pop(i))

                    self.specifiedAttList.append(attRef)
                    cl.specifiedAttList.remove(attRef)
                    changed = True
            else:
                # Continuous Attribute
                if attributeInfoType:
                    i_cl1 = self.specifiedAttList.index(attRef)
                    i_cl2 = cl.specifiedAttList.index(attRef)
                    tempKey = random.randint(0, 3)
                    if tempKey == 0:
                        temp = self.condition[i_cl1][0]
                        self.condition[i_cl1][0] = cl.condition[i_cl2][0]
                        cl.condition[i_cl2][0] = temp
                    elif tempKey == 1:
                        temp = self.condition[i_cl1][1]
                        self.condition[i_cl1][1] = cl.condition[i_cl2][1]
                        cl.condition[i_cl2][1] = temp
                    else:
                        allList = self.condition[i_cl1] + cl.condition[i_cl2]
                        newMin = min(allList)
                        newMax = max(allList)
                        if tempKey == 2:
                            self.condition[i_cl1] = [newMin, newMax]
                            cl.condition.pop(i_cl2)

                            cl.specifiedAttList.remove(attRef)
                        else:
                            cl.condition[i_cl2] = [newMin, newMax]
                            self.condition.pop(i_cl1)

                            self.specifiedAttList.remove(attRef)

                # Discrete Attribute
                else:
                    pass

        #Specification Limit Check
        if len(self.specifiedAttList) > model.rule_specificity_limit:
            self.specLimitFix(model,self)
        if len(cl.specifiedAttList) > model.rule_specificity_limit:
            self.specLimitFix(model,cl)

        tempList1 = copy.deepcopy(p_self_specifiedAttList)
        tempList2 = copy.deepcopy(cl.specifiedAttList)
        tempList1.sort()
        tempList2.sort()
        if changed and (tempList1 == tempList2):
            changed = False
        return changed

    def specLimitFix(self, model, cl):
        """ Lowers classifier specificity to specificity limit. """
        if model.do_attribute_feedback:
            # Identify 'toRemove' attributes with lowest AT scores
            while len(cl.specifiedAttList) > model.rule_specificity_limit:
                minVal = model.AT.getTrackProb()[cl.specifiedAttList[0]]
                minAtt = cl.specifiedAttList[0]
                for j in cl.specifiedAttList:
                    if model.AT.getTrackProb()[j] < minVal:
                        minVal = model.AT.getTrackProb()[j]
                        minAtt = j
                i = cl.specifiedAttList.index(minAtt)  # reference to the position of the attribute in the rule representation
                cl.specifiedAttList.remove(minAtt)
                cl.condition.pop(i)  # buildMatch handles both discrete and continuous attributes

        else:
            # Randomly pick 'toRemove'attributes to be generalized
            toRemove = len(cl.specifiedAttList) - model.rule_specificity_limit
            genTarget = random.sample(cl.specifiedAttList, toRemove)
            for j in genTarget:
                i = cl.specifiedAttList.index(j)  # reference to the position of the attribute in the rule representation
                cl.specifiedAttList.remove(j)
                cl.condition.pop(i)  # buildMatch handles both discrete and continuous attributes

    def setAccuracy(self, acc):
        """ Sets the accuracy of the classifier """
        self.accuracy = acc

    def setFitness(self, fit):
        """  Sets the fitness of the classifier. """
        self.fitness = fit

    def mutation(self,model,state):
        """ Mutates the condition of the classifier. Also handles phenotype mutation. This is a niche mutation, which means that the resulting classifier will still match the current instance.  """
        pressureProb = 0.5  # Probability that if EK is activated, it will be applied.
        useAT = model.do_attribute_feedback and random.random() < model.AT.percent
        changed = False

        steps = 0
        keepGoing = True
        while keepGoing:
            if random.random() < model.mu:
                steps += 1
            else:
                keepGoing = False

        # Define Spec Limits
        if (len(self.specifiedAttList) - steps) <= 1:
            lowLim = 1
        else:
            lowLim = len(self.specifiedAttList) - steps
        if (len(self.specifiedAttList) + steps) >= model.rule_specificity_limit:
            highLim = model.rule_specificity_limit
        else:
            highLim = len(self.specifiedAttList) + steps
        if len(self.specifiedAttList) == 0:
            highLim = 1

        # Get new rule specificity.
        newRuleSpec = random.randint(lowLim, highLim)

        # MAINTAIN SPECIFICITY
        if newRuleSpec == len(self.specifiedAttList) and random.random() < (1 - model.mu):
            #Remove random condition element
            if not model.doExpertKnowledge or random.random() > pressureProb:
                genTarget = random.sample(self.specifiedAttList,1)
            else:
                genTarget = self.selectGeneralizeRW(model,1)

            attributeInfoType = model.env.formatData.attributeInfoType[genTarget[0]]
            if not attributeInfoType or random.random() > 0.5:
                if not useAT or random.random() > model.AT.getTrackProb()[genTarget[0]]:
                    # Generalize Target
                    i = self.specifiedAttList.index(genTarget[0])  # reference to the position of the attribute in the rule representation
                    self.specifiedAttList.remove(genTarget[0])
                    self.condition.pop(i)  # buildMatch handles both discrete and continuous attributes
                    changed = True
            else:
                self.mutateContinuousAttributes(model,useAT, genTarget[0])

            #Add random condition element
            if len(self.specifiedAttList) >= len(state):
                pass
            else:
                if not model.doExpertKnowledge or random.random() > pressureProb:
                    pickList = list(range(model.env.formatData.numAttributes))
                    for i in self.specifiedAttList:
                        pickList.remove(i)
                    specTarget = random.sample(pickList,1)
                else:
                    specTarget = self.selectSpecifyRW(model,1)

                if state[specTarget[0]] != None and (not useAT or random.random() < model.AT.getTrackProb()[specTarget[0]]):
                    self.specifiedAttList.append(specTarget[0])
                    self.condition.append(self.buildMatch(model,specTarget[0],state))  # buildMatch handles both discrete and continuous attributes
                    changed = True
                if len(self.specifiedAttList) > model.rule_specificity_limit:
                    self.specLimitFix(model,self)

        #Increase Specificity
        elif newRuleSpec > len(self.specifiedAttList): #Specify more attributes
            change = newRuleSpec - len(self.specifiedAttList)
            if not model.doExpertKnowledge or random.random() > pressureProb:
                pickList = list(range(model.env.formatData.numAttributes))
                for i in self.specifiedAttList: # Make list with all non-specified attributes
                    pickList.remove(i)
                specTarget = random.sample(pickList,change)
            else:
                specTarget = self.selectSpecifyRW(model,change)
            for j in specTarget:
                if state[j] != None and (not useAT or random.random() < model.AT.getTrackProb()[j]):
                    #Specify Target
                    self.specifiedAttList.append(j)
                    self.condition.append(self.buildMatch(model,j, state)) #buildMatch handles both discrete and continuous attributes
                    changed = True

        #Decrease Specificity
        elif newRuleSpec < len(self.specifiedAttList): # Generalize more attributes.
            change = len(self.specifiedAttList) - newRuleSpec
            if not model.doExpertKnowledge or random.random() > pressureProb:
                genTarget = random.sample(self.specifiedAttList,change)
            else:
                genTarget = self.selectGeneralizeRW(model,change)

            #-------------------------------------------------------
            # DISCRETE OR CONTINUOUS ATTRIBUTE - remove attribute specification with 50% chance if we have continuous attribute, or 100% if discrete attribute.
            #-------------------------------------------------------
            for j in genTarget:
                attributeInfoType = model.env.formatData.attributeInfoType[j]
                if not attributeInfoType or random.random() > 0.5: #GEN/SPEC OPTION
                    if not useAT or random.random() > model.AT.getTrackProb()[j]:
                        i = self.specifiedAttList.index(j) #reference to the position of the attribute in the rule representation
                        self.specifiedAttList.remove(j)
                        self.condition.pop(i) #buildMatch handles both discrete and continuous attributes
                        changed = True
                else:
                    self.mutateContinuousAttributes(model,useAT,j)

        return changed

    def selectGeneralizeRW(self,model,count):
        probList = []
        for attribute in self.specifiedAttList:
            probList.append(1/model.EK.scores[attribute])
        if sum(probList) == 0:
            probList = (np.array(probList) + 1).tolist()

        probList = np.array(probList)/sum(probList) #normalize
        return np.random.choice(self.specifiedAttList,count,replace=False,p=probList).tolist()

    # def selectGeneralizeRW(self,model,count):
    #     EKScoreSum = 0
    #     selectList = []
    #     currentCount = 0
    #     specAttList = copy.deepcopy(self.specifiedAttList)
    #     for i in self.specifiedAttList:
    #         # When generalizing, EK is inversely proportional to selection probability
    #         EKScoreSum += 1 / float(model.EK.scores[i] + 1)
    #
    #     while currentCount < count:
    #         choicePoint = random.random() * EKScoreSum
    #         i = 0
    #         sumScore = 1 / float(model.EK.scores[specAttList[i]] + 1)
    #         while choicePoint > sumScore:
    #             i = i + 1
    #             sumScore += 1 / float(model.EK.scores[specAttList[i]] + 1)
    #         selectList.append(specAttList[i])
    #         EKScoreSum -= 1 / float(model.EK.scores[specAttList[i]] + 1)
    #         specAttList.pop(i)
    #         currentCount += 1
    #     return selectList

    def selectSpecifyRW(self,model,count):
        pickList = list(range(model.env.formatData.numAttributes))
        for i in self.specifiedAttList:  # Make list with all non-specified attributes
            pickList.remove(i)

        probList = []
        for attribute in pickList:
            probList.append(model.EK.scores[attribute])
        if sum(probList) == 0:
            probList = (np.array(probList) + 1).tolist()
        probList = np.array(probList) / sum(probList)  # normalize
        return np.random.choice(pickList, count, replace=False, p=probList).tolist()

    # def selectSpecifyRW(self, model,count):
    #     """ EK applied to the selection of an attribute to specify for mutation. """
    #     pickList = list(range(model.env.formatData.numAttributes))
    #     for i in self.specifiedAttList:  # Make list with all non-specified attributes
    #         pickList.remove(i)
    #
    #     EKScoreSum = 0
    #     selectList = []
    #     currentCount = 0
    #
    #     for i in pickList:
    #         # When generalizing, EK is inversely proportional to selection probability
    #         EKScoreSum += model.EK.scores[i]
    #
    #     while currentCount < count:
    #         choicePoint = random.random() * EKScoreSum
    #         i = 0
    #         sumScore = model.EK.scores[pickList[i]]
    #         while choicePoint > sumScore:
    #             i = i + 1
    #             sumScore += model.EK.scores[pickList[i]]
    #         selectList.append(pickList[i])
    #         EKScoreSum -= model.EK.scores[pickList[i]]
    #         pickList.pop(i)
    #         currentCount += 1
    #     return selectList

    def mutateContinuousAttributes(self, model,useAT, j):
        # -------------------------------------------------------
        # MUTATE CONTINUOUS ATTRIBUTES
        # -------------------------------------------------------
        if useAT:
            if random.random() < model.AT.getTrackProb()[j]:  # High AT probability leads to higher chance of mutation (Dives ExSTraCS to explore new continuous ranges for important attributes)
                # Mutate continuous range - based on Bacardit 2009 - Select one bound with uniform probability and add or subtract a randomly generated offset to bound, of size between 0 and 50% of att domain.
                attRange = float(model.env.formatData.attributeInfoContinuous[j][1]) - float(model.env.formatData.attributeInfoContinuous[j][0])
                i = self.specifiedAttList.index(j)  # reference to the position of the attribute in the rule representation
                mutateRange = random.random() * 0.5 * attRange
                if random.random() > 0.5:  # Mutate minimum
                    if random.random() > 0.5:  # Add
                        self.condition[i][0] += mutateRange
                    else:  # Subtract
                        self.condition[i][0] -= mutateRange
                else:  # Mutate maximum
                    if random.random() > 0.5:  # Add
                        self.condition[i][1] += mutateRange
                    else:  # Subtract
                        self.condition[i][1] -= mutateRange
                # Repair range - such that min specified first, and max second.
                self.condition[i].sort()
                changed = True
        elif random.random() > 0.5:
            # Mutate continuous range - based on Bacardit 2009 - Select one bound with uniform probability and add or subtract a randomly generated offset to bound, of size between 0 and 50% of att domain.
            attRange = float(model.env.formatData.attributeInfoContinuous[j][1]) - float(model.env.formatData.attributeInfoContinuous[j][0])
            i = self.specifiedAttList.index(j)  # reference to the position of the attribute in the rule representation
            mutateRange = random.random() * 0.5 * attRange
            if random.random() > 0.5:  # Mutate minimum
                if random.random() > 0.5:  # Add
                    self.condition[i][0] += mutateRange
                else:  # Subtract
                    self.condition[i][0] -= mutateRange
            else:  # Mutate maximum
                if random.random() > 0.5:  # Add
                    self.condition[i][1] += mutateRange
                else:  # Subtract
                    self.condition[i][1] -= mutateRange
            # Repair range - such that min specified first, and max second.
            self.condition[i].sort()
            changed = True
        else:
            pass


    def rangeCheck(self,model):
        """ Checks and prevents the scenario where a continuous attributes specified in a rule has a range that fully encloses the training set range for that attribute."""
        for attRef in self.specifiedAttList:
            if model.env.formatData.attributeInfoType[attRef]: #Attribute is Continuous
                trueMin = model.env.formatData.attributeInfoContinuous[attRef][0]
                trueMax = model.env.formatData.attributeInfoContinuous[attRef][1]
                i = self.specifiedAttList.index(attRef)
                valBuffer = (trueMax-trueMin)*0.1
                if self.condition[i][0] <= trueMin and self.condition[i][1] >= trueMax: # Rule range encloses entire training range
                    self.specifiedAttList.remove(attRef)
                    self.condition.pop(i)
                    return
                elif self.condition[i][0]+valBuffer < trueMin:
                    self.condition[i][0] = trueMin - valBuffer
                elif self.condition[i][1]- valBuffer > trueMax:
                    self.condition[i][1] = trueMin + valBuffer
                else:
                    pass

    def getDelProp(self, model, meanFitness):
        """  Returns the vote for deletion of the classifier. """
        if self.fitness / self.numerosity >= model.delta * meanFitness or self.matchCount < model.theta_del:
            deletionVote = self.aveMatchSetSize * self.numerosity
        elif self.fitness == 0.0:
            deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (model.init_fitness / self.numerosity)
        else:
            deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (self.fitness / self.numerosity)
        return deletionVote
