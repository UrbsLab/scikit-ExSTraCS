
class SURFStar:
    def __init__(self,model):
        data = model.env.formatData
        dataFeatures = data.trainFormatted[0]
        dataPhenotypes = data.trainFormatted[1]
        maxInst = int(float(model.reliefSampleFraction)*len(dataFeatures))
        return self.run(dataFeatures,dataPhenotypes,maxInst,data)

    def run(self,features,phenotypes,maxInst,data):
        scoreList = []
        for i in range(data.numAttributes):
            scoreList.append(0)

        distanceObject = self.calculateDistanceArray(features, data, maxInst)
        distanceArray = distanceObject[0]
        averageDistance = distanceObject[1]

    def calculateDistanceArray(self,x, data, maxInst):
        """ In SURFStar this method precomputes both the distance array and the average distance """
        # make empty distance array container (we will only fill up the non redundant half of the array
        distArray = []
        aveDist = 0
        count = 0
        for i in range(maxInst):
            distArray.append([])
            for j in range(maxInst):
                distArray[i].append(None)

        for i in range(1, maxInst):
            for j in range(0, i):
                distArray[i][j] = self.calculate_distance(x[i], x[j], data)
                count += 1
                aveDist += distArray[i][j]

        aveDist = aveDist / float(count)
        returnObject = [distArray, aveDist]
        return returnObject

    def calculate_distance(self,a, b, data):
        """ Calculates the distance between two instances in the dataset.  Handles discrete and continuous attributes. Continuous attributes are accomodated
        by scaling the distance difference within the context of the observed attribute range. If a respective data point is missing from either instance, it is left out
        of the distance calculation. """
        d = 0  # distance
        for i in range(data.numAttributes):
            if a[i] != None and b[i] != None:
                if not data.attributeInfoType[i]:  # Discrete Attribute
                    if a[i] != b[i]:
                        d += 1
                else:  # Continuous Attribute
                    min_bound = float(data.attributeInfoContinuous[i][0])
                    max_bound = float(data.attributeInfoContinuous[i][1])
                    d += abs(float(a[i]) - float(b[i])) / float(
                        max_bound - min_bound)  # Kira & Rendell, 1992 -handling continiuous attributes
        return d
