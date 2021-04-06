#-------------------------------------------------------------------------
# AUTHOR: Daniel Molina
# FILENAME: svm.py
# SPECIFICATION: read the file optdigits.tra to build multiple SVM classifiers
# FOR: CS 4200- Assignment #3
# TIME SPENT: 40 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import svm
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
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
highestAccuracy = 0

#reading the data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
  reader = csv.reader(trainingFile)
  for i, row in enumerate(reader):
      X_training.append(row[:-1])
      Y_training.append(row[-1:])

#reading the data in a csv file
with open('optdigits.tes', 'r') as testingFile:
  reader = csv.reader(testingFile)
  for i, row in enumerate(reader):
      dbTest.append (row)

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
accuracyMax = 0
c_max = 0
d_max = 0
k_max = ""
shape_max = ""

for c_i in c: #iterates over c
    for d in degree: #iterates over degree
        for k in kernel: #iterates kernel
           for shape in decision_function_shape: #iterates over decision_function_shape

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape as hyperparameters. For instance svm.SVC(c=1)
                clf = svm.SVC(decision_function_shape=shape, kernel=k, degree=d, C=c_i)

                #Fit Random Forest to the training data
                clf.fit(X_training, Y_training)

                #make the classifier prediction for each test sample and start computing its accuracy
                #--> add your Python code here
                correctClassifications = 0
                for testSample in dbTest:
                    class_predicted = clf.predict([testSample[:-1]])[0]
                    if(class_predicted == testSample[-1]):
                        correctClassifications += 1
                
                accuracy = correctClassifications / len(dbTest)

                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together with the SVM hyperparameters
                #Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here
                if(accuracy > accuracyMax):
                    print("Highest SVM accuracy so far: %.4f, Parameters: c=%d, degree=%d, kernel= %s, decision_function_shape = %s"%(accuracy, c_i,d,k,shape))
                    accuracyMax = accuracy
                    c_max = c_i
                    d_max = d
                    k_max = k
                    shape_max = shape
#print the final, highest accuracy found together with the SVM hyperparameters
#Example: "Highest SVM accuracy: 0.95, Parameters: a=10, degree=3, kernel= poly, decision_function_shape = 'ovr'"
print("finished checking all SVM classifiers, the best one for this instance is.....")
print("Highest SVM accuracy: %.4f, Parameters: c=%d, degree=%d, kernel= %s, decision_function_shape = %s"%(accuracyMax, c_max,d_max,k_max,shape_max))

