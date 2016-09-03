# K nearset neighbors on iris dataset.
import csv
import random
import math



#read data from 'iris.data' and parse it
def parseDataSet(fileName, trainingData, testData, trainFraction):
	datafile = open(fileName, 'rb')
	content = csv.reader(datafile)
	dataset = list(content)

	for i in range(len(dataset)):
		for j in range(len(dataset[0])-1):
			dataset[i][j] = float(dataset[i][j])
			if random.randint(1,100) < trainFraction:
				trainingData.append(dataset[i])
			else:
				testData.append(dataset[i])




#since the values are in same units, normal euclidian distance can be used as difference measure between two instances
def diffFactor(instance1, instance2, numOfAtt):
	difference = 0;
	for i in range(numOfAtt):
		difference += pow(instance1[i] - instance2[i], 2)

	return math.sqrt(difference)




#getKey for sort
def getKey(item):
	return item[1]

#find k nearest 'k' neighbors to a given test case
def findKNeighbors(trainingData, testCase, k):
	allDiff = []
	for i in range(len(trainingData)):
		diff = diffFactor(trainingData[i], testCase, 4)
		allDiff.append((trainingData[i], diff))

	allDiff.sort(key=getKey)
	knn = []
	for i in range(k):
		knn.append(allDiff[i][0][4])

	return knn




def finalDecision(knn):
	votes = {}
	for i in range(len(knn)):
		if knn[i] in votes:
			votes[knn[i]] += 1
		else:
			votes[knn[i]] = 1

	sortedVotes = sorted(votes.keys())
	return sortedVotes[0]


def accuracy(testData, ourClassification):
	correct = 0
	for i in range(len(testData)):
		if testData[i][4] == ourClassification[i]:
			correct += 1

	return (correct/float(len(testData)))*100



def main():

	trainingData = []
	testData = []
	parseDataSet('iris.data', trainingData, testData, 30)

	ourClassification = []
	for i in range(len(testData)):
		testCase = testData[i]
		knn = findKNeighbors(trainingData, testCase, 10)
		decision = finalDecision(knn)
		if testData[i][4] != decision:
			print 'predicted: ' + str(testData[i][4]) + '    groundTruth: ' + str(decision)
		ourClassification.append(decision)

	acc = accuracy(testData, ourClassification)
	print 'Accuracy: ' + str(acc) + '%'

main()