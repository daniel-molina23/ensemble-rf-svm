#-------------------------------------------------------------------------
# AUTHOR: Daniel Molina
# FILENAME: bagging_random_forest.py
# SPECIFICATION: build a base classifier by using a single decision tree, 
#       then build an ensemble classifier that combines multiple
#       decision trees, and lastly use a Random Forest classifier for comparison.
# FOR: CS 4200- Assignment #3
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import csv

# --------------------- Makes for better looking output ---------------
# ignores dataConversion Warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# ---------------------------------------------------------------------


dbTraining = []
dbTest = []
X_training = []
Y_training = []
classVotes = [] #this array will be used to count the votes of each classifier

#reading the training data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
    reader = csv.reader(trainingFile)
    for i, row in enumerate(reader):
        dbTraining.append (row)


#reading the test data in a csv file
with open('optdigits.tes', 'r') as testingFile:
    reader = csv.reader(testingFile)
    for i, row in enumerate(reader):
        dbTest.append (row)
        classVotes.append([0,0,0,0,0,0,0,0,0,0]) #inititalizing the class votes for each test sample

print("Started my base and ensemble classifier ...")


for k in range(20): #we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample

    bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)

    #populate the values of X_training and Y_training by using the bootstrapSample
    X_training = []
    Y_training = []
    for row in bootstrapSample:
        X_training.append(row[:-1])
        Y_training.append([row[-1]])

    #fitting the decision tree to the data
    clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=None) #we will use a single decision tree without pruning it
    clf = clf.fit(X_training, Y_training)

    correctNum = 0
    for i, testSample in enumerate(dbTest):

        #make the classifier prediction for each test sample and update the corresponding index value in classVotes. For instance,
        # if your first base classifier predicted 2 for the first test sample, then classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0].
        # Later, if your second base classifier predicted 3 for the first test sample, then classVotes[0,0,1,0,0,0,0,0,0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0]
        # Later, if your third base classifier predicted 3 for the first test sample, then classVotes[0,0,1,1,0,0,0,0,0,0] will change to classVotes[0,0,1,2,0,0,0,0,0,0]
        # this arrays will consolidate the votes of all classifier for all test samples
        y_actual = dbTest[i][-1]
        y_pred = clf.predict([dbTest[i][:-1]])[0]
        classVotes[i][int(y_pred)] += 1 # increment the vote for the ith subset at the index the Decision Tree thinks is correct
        

        if (k == 0): 
            #for only the first base classifier, compare the prediction with the true label
            #  of the test sample here to start calculating its accuracy
            if(y_pred == y_actual):
                correctNum += 1
    # end for loop
    if (k == 0): #for only the first base classifier, print its accuracy here
        accuracy = correctNum / len(dbTest)
        print("Finished my base classifier (fast but relatively low accuracy) ...")
        print("My base classifier accuracy: " + str(accuracy))
        print("")

#now, compare the final ensemble prediction (majority vote in classVotes) for each test sample with the ground truth label to calculate the accuracy of the ensemble classifier (all base classifiers together)
correctClassifications = 0
for i,row in enumerate(classVotes):
    ensemblePred = "-1"
    maxNum = -1
    
    for j, count in enumerate(row):
        if(count > maxNum):
            ensemblePred = str(j)
            maxNum = count
    
    if(ensemblePred == dbTest[i][-1]):
        correctClassifications += 1

accuracy = correctClassifications / len(dbTest)

#printing the ensemble accuracy here
print("Finished my ensemble classifier (slow but higher accuracy) ...")
print("My ensemble accuracy: " + str(accuracy))
print("")

print("Started Random Forest algorithm ...")

#Create a Random Forest Classifier
clf=RandomForestClassifier(n_estimators=20) #this is the number of decision trees that will be generated by Random Forest. The sample of the ensemble method used before

# create new X_train and Y_train
X_training = []
Y_training = []
for row in dbTraining:
    X_training.append(row[:-1])
    Y_training.append([row[-1]])


#Fit Random Forest to the training data
clf.fit(X_training,Y_training)

correctClassifications = 0
#make the Random Forest prediction for each test sample. Example: class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
for testInstance in dbTest:
    y_actual = testInstance[-1]
    y_pred = clf.predict([testInstance[:-1]])[0]
    
    #compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
    if(y_actual == y_pred):
        correctClassifications += 1

accuracy = correctClassifications / len(dbTest)

#printing Random Forest accuracy here
print("Random Forest accuracy: " + str(accuracy))

print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
