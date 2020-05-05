
class MultiSURF:
    def __init__(self,model):
        data = model.env.formatData
        dataFeatures = data.trainFormatted[0]
        dataPhenotypes = data.trainFormatted[1]
        return self.run(dataFeatures,dataPhenotypes,data)

    def run(self,features,phenotypes,data):
        scoreList = []
        for i in range(data.numAttributes):
            scoreList.append(0)
        distanceArray = self.calculateDistanceArray(features,data)

        D = []
        avg_distances = []
        for i in range(data.numTrainInstances):
            dist_vector = self.get_individual_distances(i, data, distanceArray)
            avg_distances.append(self.get_average(dist_vector))
            std_dev = self.get_std_dev(dist_vector, avg_distances[i])
            D.append(std_dev / 2.0)

        for k in range(data.numAttributes):
            if data.attributeInfoType[k]: #Continuous Attributes
                minA = data.attributeInfoContinuous[k][0]
                maxA = data.attributeInfoContinuous[k][1]

            count_hit_near = 0
            count_miss_near = 0
            count_hit_far = 0
            count_miss_far = 0

            diff_hit_near = 0  # initializing the score to 0
            diff_miss_near = 0
            diff_hit_far = 0
            diff_miss_far = 0

            for i in range(data.numTrainInstances):
                for j in range(i, data.numTrainInstances):
                    if i != j and features[i][k] != None and features[j][k] != None:
                        locator = [i, j]
                        locator = sorted(locator,reverse=True)  # Access correct half of table (result of removed table redundancy)
                        d = distanceArray[locator[0]][locator[1]]

                        if (d < avg_distances[i] - D[i]):  # Near
                            if phenotypes[i] == phenotypes[j]:  # Same Endpoint
                                count_hit_near += 1
                                if features[i][k] != features[j][k]:
                                    if data.attributeInfoType[k]:  # Continuous Attribute (closer att scores for near same phen should yield larger att penalty)
                                        diff_hit_near -= (abs(features[i][k] - features[j][k]) / (maxA - minA))
                                    else:  # Discrete
                                        diff_hit_near -= 1
                            else:  # Different Endpoint
                                count_miss_near += 1
                                if features[i][k] != features[j][k]:
                                    if data.attributeInfoType[k]:  # Continuous Attribute (farther att scores for near diff phen should yield larger att bonus)
                                        diff_miss_near += abs(features[i][k] - features[j][k]) / (maxA - minA)
                                    else:  # Discrete
                                        diff_miss_near += 1

                        if (d > avg_distances[i] + D[i]):  # Far
                            if phenotypes[i] == phenotypes[j]:
                                count_hit_far += 1
                                if data.attributeInfoType[k]:  # Continuous Attribute
                                    diff_hit_far -= (abs(features[i][k] - features[j][k])) / (maxA - minA)  # Attribute being similar is more important.
                                else:  # Discrete
                                    if features[i][k] == features[j][k]:
                                        diff_hit_far -= 1
                            else:
                                count_miss_far += 1
                                if data.attributeInfoType[k]:  # Continuous Attribute
                                    diff_miss_far += abs(features[i][k] - features[j][k]) / (maxA - minA)  # Attribute being similar is more important.
                                else:  # Discrete
                                    if features[i][k] == features[j][k]:
                                        diff_miss_far += 1

            hit_proportion = count_hit_near / float(count_hit_near + count_miss_near)
            miss_proportion = count_miss_near / float(count_hit_near + count_miss_near)

            diff = diff_hit_near * miss_proportion + diff_miss_near * hit_proportion  # applying weighting scheme to balance the scores

            hit_proportion = count_hit_far / float(count_hit_far + count_miss_far)
            miss_proportion = count_miss_far / float(count_hit_far + count_miss_far)

            diff += diff_hit_far * miss_proportion + diff_miss_far * hit_proportion  # applying weighting scheme to balance the scores

            scoreList[k] += diff
        return scoreList

    def get_std_dev(self,dist_vector, avg):
        sum = 0;
        for i in range(len(dist_vector)):
            sum += (dist_vector[i] - avg) ** 2
        sum = sum / float(len(dist_vector))
        return (sum ** 0.5)

    def get_average(self,dist_vector):
        sum = 0
        for i in range(len(dist_vector)):
            sum += dist_vector[i];
        return sum / float(len(dist_vector))

    def get_individual_distances(self,i, data, distanceArray):
        d = []
        for j in range(data.numTrainInstances):
            if (i != j):
                locator = [i, j]
                locator = sorted(locator,
                                 reverse=True)  # Access corect half of table (result of removed table redundancy)
                d.append(distanceArray[locator[0]][locator[1]])
        return d

    def calculateDistanceArray(self,features,data):
        distArray = []
        for i in range(data.numTrainInstances):
            distArray.append([])
            for j in range(data.numTrainInstances):
                distArray[i].append(None)

        for i in range(1, data.numTrainInstances):
            for j in range(0, i):
                distArray[i][j] = self.calculate_distance(features[i], features[j], data)

        return distArray

    def calculateDistance(self,a,b,data):
        d = 0  # distance
        for i in range(data.numAttributes):
            if a[i] != None and b[i] != None:
                if not data.attributeInfoType[i]:  # Discrete Attribute
                    if a[i] != b[i]:
                        d += 1
                else:  # Continuous Attribute
                    min_bound = float(data.attributeInfoContinuous[i][0])
                    max_bound = float(data.attributeInfoContinuous[i][1])
                    d += abs(float(a[i]) - float(b[i])) / float(max_bound - min_bound)  # Kira & Rendell, 1992 -handling continiuous attributes
        return d