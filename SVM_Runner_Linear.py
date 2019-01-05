import random
import numpy as np
from PIL import Image
from cvxopt import solvers, matrix

# first divide images into 5 subsets

setValues = random.sample(range(1,11),10)
set1 = setValues[0:2]
set2 = setValues[2:4]
set3 = setValues[4:6]
set4 = setValues[6:8]
set5 = setValues[8:10]

accuracy = 0

# choose and assign test set images to testImages and train set images to trainImages
for testSet in range(1,6): 
	testImages = [None] * 2
	trainImages = [None] * 8
	xVals = np.zeros((16,92*112))
	yVals = np.zeros(16)
	Ws = np.zeros((40,40,10304))
	biases = np.zeros((40,40))

	bias = 0
	j = 1

	if (testSet == 1): 
		testImages[0] = str(set1[0]) + ".pgm"
		testImages[1] = str(set1[1]) + ".pgm"
		for i in range(0,8):
			if(j == set1[0] or j == set1[1]):
				j += 1
			if(j == set1[0] or j == set1[1]):
				j += 1
			trainImages[i] = str(j) + ".pgm"
			j += 1
	elif (testSet == 2):	
		testImages[0] = str(set2[0]) + ".pgm"
		testImages[1] = str(set2[1]) + ".pgm"
		for i in range(0,8):
			if(j == set2[0] or j == set2[1]):
				j += 1
			if(j == set2[0] or j == set2[1]):
				j += 1
			trainImages[i] = str(j) + ".pgm"
			j += 1
	elif (testSet == 3):
		testImages[0] = str(set3[0]) + ".pgm"
		testImages[1] = str(set3[1]) + ".pgm"
		for i in range(0,8):
			if(j == set3[0] or j == set3[1]):
				j += 1
			if(j == set3[0] or j == set3[1]):
				j += 1
			trainImages[i] = str(j) + ".pgm"
			j += 1
	elif (testSet == 4):
		testImages[0] = str(set4[0]) + ".pgm"
		testImages[1] = str(set4[1]) + ".pgm"
		for i in range(0,8):
			if(j == set4[0] or j == set4[1]):
				j += 1
			if(j == set4[0] or j == set4[1]):
				j += 1
			trainImages[i] = str(j) + ".pgm"
			j += 1
	else:	
		testImages[0] = str(set5[0]) + ".pgm"
		testImages[1] = str(set5[1]) + ".pgm"
		for i in range(0,8):
			if(j == set5[0] or j == set5[1]):
				j += 1
			if(j == set5[0] or j == set5[1]):
				j += 1
			trainImages[i] = str(j) + ".pgm"
			j += 1

	# train 860 classifiers

	for class1 in range(1, 40):
		c1 = "faces/s" + str(class1)
		for class2 in range(class1+1, 41):
			c2 = "faces/s" + str(class2)
			for i in range(0, 8):
				im1 = Image.open(c1 + "/" + trainImages[i], 'r')
				pix_vals_1 = list(im1.getdata())
				im2 = Image.open(c2 + "/" + trainImages[i], 'r')
				pix_vals_2 = list(im2.getdata())
				
				xVals[i] = pix_vals_1
				xVals[i+8] = pix_vals_2
				yVals[i] = 1
				yVals[i+8] = -1
			
			# calculate all matrices needed for the qp solver
			K = yVals[:, None] * xVals
			K = np.dot(K, K.T)
			P = matrix(K)
			q = matrix(-np.ones((16, 1)))
			G = matrix(-np.eye(16))
			h = matrix(np.zeros(16))
			A = matrix(yVals.reshape(1, -1))
			b = matrix(np.zeros(1))
			
			solvers.options['show_progress'] = False
			sol = solvers.qp(P, q, G, h, A, b)
			alphas = np.array(sol['x'])
			
			w = np.sum(alphas * yVals[:, None] * xVals, axis = 0)
			
			for index in range(0, 16):
				if (alphas[index] > 1e-8):
					bias = yVals[index] - np.dot(xVals[index], w)
					break
			
			# calculate and assign slopes and intercepts of divider lines from training data
			Ws[class1-1][class2-1] = w
			biases[class1-1][class2-1] = bias
				
	# test

	numTested = 0
	numCorrect = 0

	for actualClass in range(1, 41):
		ac = "faces/s" + str(actualClass)
		faceCounters1 = np.zeros(40)
		faceCounters2 = np.zeros(40)
			
		testImg1 = Image.open(ac + '/' + testImages[0])
		pix_vals_t1 = list(testImg1.getdata())
		
		testImg2 = Image.open(ac + '/' + testImages[1])
		pix_vals_t2 = list(testImg2.getdata())
		
		for class1 in range(1, 40):	
			for class2 in range(class1 + 1, 41):
				if((np.dot(pix_vals_t1, Ws[class1-1][class2-1]) + biases[class1-1][class2-1]) >= 0):
					faceCounters1[class1-1] += 1
				else:
					faceCounters1[class2-1] += 1
					
				if((np.dot(pix_vals_t2, Ws[class1-1][class2-1]) + biases[class1-1][class2-1]) >= 0):
					faceCounters2[class1-1] += 1
				else:
					faceCounters2[class2-1] += 1
		
		# update number of tested images and number of properly classified images accordingly
		numTested += 2
		if(np.argmax(faceCounters1)+1 == actualClass):
			numCorrect += 1
		if(np.argmax(faceCounters2)+1 == actualClass):
			numCorrect += 1
		
	# evaluate
				
	print ("Your accuracy for linear test number " + str(testSet) + " was " + str(numCorrect/numTested))
	accuracy += numCorrect/numTested*100

print("Your average linear test accuracy was: " + str(accuracy/5) + "%")
