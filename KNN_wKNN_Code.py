#********************************************************************************
# Author - Ritam Das
#********************************************************************************

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#********************************************************************************
# Global Constants
#********************************************************************************
SCALE_MINMAX, SCALE_ZNORMAL = 'minMax', 'zNormal'
DIST_EUCLIDEAN, DIST_MANHATTAN = 'Euclidean','Manhattan'

#********************************************************************************
# Extracts necessary data, returns dataframe
#********************************************************************************
def getFeatureSet(data_frame):
    return data_frame[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides","free sulfur dioxide",
                       "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]

def getLabelSet(data_frame):
    return data_frame["Quality"]
    
#********************************************************************************
# Calculates distances - Euclidean and Manhattan
#********************************************************************************
def calculateManhattanDistances(dataSet, query_point):
    query_point = query_point.to_numpy()
    dataSet = dataSet.to_numpy()
    query_point_repeat = np.repeat(query_point,dataSet.shape[0],axis=0)
    dist = np.sum(np.abs(dataSet - query_point_repeat),axis=1)
    return dist

def calculateDistance(dataSet, query_point):
    query_point = query_point.to_numpy()
    query_point_repeat = np.repeat(query_point,dataSet.shape[0],axis=0)
    dist = np.linalg.norm(dataSet - query_point_repeat,axis=1)
    return dist
    
#********************************************************************************
# Normalizes and standardizes data
#In normalizing the testing data, scaling standards (min, max / standard deviation,mean) are taken from training data.
#Being unsure of a query point's quality it is done to prevent erroneous scaling. 
#Additionally, if test data consists of a single query point scaling standards cannot be determined.
#********************************************************************************    
def minMaxNormalization(test_feature, train_feature): 
    #min, max are taken from training data. Reason described above.   
    scaled_train = (train_feature - train_feature.min(axis = 0)) / (train_feature.max(axis = 0) - train_feature.min(axis = 0))
    scaled_test = (test_feature - train_feature.min(axis = 0)) / (train_feature.max(axis = 0) - train_feature.min(axis = 0))
    return scaled_test,scaled_train
    
def zNormalization(test_feature, train_feature):
    #standard deviation, mean are taken from training data. Reason described above. 
    scaled_train = (train_feature - train_feature.mean(axis = 0))  / (np.std(train_feature.to_numpy(), axis = 0 , dtype = np.float64))
    scaled_test = (test_feature - train_feature.mean(axis = 0))  / (np.std(train_feature.to_numpy(), axis = 0 , dtype = np.float64))
    return scaled_test,scaled_train

#********************************************************************************
# KNN - Calculates accuracy for both weighted and unweighted
#********************************************************************************    
def knn(k,scaling,distance):
    # read the dataset training and test data
    df_train = pd.read_csv("wine-data-project-train.csv")
    df_test  = pd.read_csv("wine-data-project-test.csv")

    #features - test dataset
    test_feature = getFeatureSet(df_test)
    test_label_Set = getLabelSet(df_test)
    
    #features - training dataset
    train_feature = getFeatureSet(df_train)
    train_label = getLabelSet(df_train)
  
    #Scale data if instructed by user
    if scaling == SCALE_MINMAX:
        test_feature, train_feature = minMaxNormalization(test_feature, train_feature)
    elif scaling == SCALE_ZNORMAL:
        test_feature, train_feature = zNormalization(test_feature, train_feature)
    elif scaling == None:
        pass
    else:
        print("Scaling not recognized")
        return
    
    # keep count of the number of correct classifications
    correctClassifications = 0

    # classify each instance from the test feature dataset in turn
    for num in range(0, test_feature.shape[0]):
        test_label_instance  = df_test["Quality"].iloc[[num]]
        
        #Distance metric set as needed
        if distance == DIST_EUCLIDEAN:
            dist = calculateDistance(train_feature,test_feature.iloc[[num]])
        elif distance == DIST_MANHATTAN:
            dist = calculateManhattanDistances(train_feature,test_feature.iloc[[num]])
        else:
            print("Distance metric not recognized")
            return
        
        #Finds k lowest distanced points and checks their corresponding labels against test
        idx = np.argpartition(dist, k)
        loc = idx[:k]
        loc = loc.tolist()  
        class_labels = list(train_label.iloc[loc,])
        predicted_class = max(set(class_labels), key = class_labels.count)
        correctClassifications += np.sum(test_label_instance.to_numpy() == predicted_class)
    accuracy = np.round((correctClassifications*100)/test_feature.shape[0], 2)
    
    #weighted KNN called with same k value
    accuracyWeightedModel = Weightedknn(k,train_feature,train_label,test_feature,test_label_Set,distance)
    
    #returns both accuracies as instructed in guidelines
    return [accuracy,accuracyWeightedModel]
 
#********************************************************************************
# Weighted KNN - returns accuracyWeightedModel
#******************************************************************************** 
def Weightedknn(k,train_feature,train_label,test_feature,test_label_Set,distance):

    # keep count of the number of correct classifications
    correctClassifications = 0

    # classify each instance from the test feature dataset in turn
    for num in range(0, test_feature.shape[0]):
        test_label_instance  = test_label_Set.iloc[[num]]
        label_list = list(np.unique(test_label_Set))
        vote = [0]*len(label_list)
        
        #Distance metric set as needed
        if distance == DIST_EUCLIDEAN:
            dist = calculateDistance(train_feature,test_feature.iloc[[num]])
        elif distance == DIST_MANHATTAN:
            dist = calculateManhattanDistances(train_feature,test_feature.iloc[[num]])
        
        #Finds k lowest distanced points and their corresponding labelspyt
        idx = np.argpartition(dist, k)
        loc = idx[:k]
        loc = loc.tolist()
        class_labels = list(train_label.iloc[loc,])
        
        #Voting begins
        for i,j in zip(class_labels,dist[loc]):
            if j != 0:
                weight = 1/j**2
            else:
                weight = 1
            #Dynamic - will be handle more than 2 classes if needed
            if i in label_list:
                vote[label_list.index(i)] += weight
        
        #Label with most votes fetched and checked against test
        predicted_class = label_list[np.argmax(vote)]
        if np.sum(predicted_class == test_label_instance.to_numpy()) == 1:
            correctClassifications += 1
    accuracy = np.round((correctClassifications*100)/test_feature.shape[0], 2)
    return accuracy

#********************************************************************************
# Main function
# Parameters: 
# scaling (accepts 'minMax', 'zNormal', None) 
# distance (accepts 'Euclidean','Manhattan')
#********************************************************************************
def main(scaling = None,distance = DIST_EUCLIDEAN):
    allResults = []
    allWeightedResults = []
    for k in range(3, 40, 2):
        accuracy = knn(k,scaling,distance)
        allResults.append(accuracy[0])
        allWeightedResults.append(accuracy[1])
    sns.set_style("darkgrid")
    plt.figure(figsize = (10, 6))
    plt.plot( list(range(3, 40, 2)), allResults, label = "KNN")
    plt.plot( list(range(3, 40, 2)), allWeightedResults, label = "Weighted KNN")
    plt.xlabel('K')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    if scaling == None:
        scaling = 'no'
    plt.title('Accuracy for '+scaling+ ' scaling and '+distance+' distance')
    plt.show()
    
#********************************************************************************
# Driver Code - Runnable
# Run as-is, without any change
# Plots both KNN and Weighted KNN accuracy graphs
# Dependencies - Global constants and all methods
#********************************************************************************
if __name__ == "__main__":
    main() #distance - Euclidean, scaling - None
    main(distance = DIST_MANHATTAN) #Distance - Manhattan, Scaling - None
    main(scaling = SCALE_MINMAX) #Distance - Euclidean, Scaling - MinMax
    main(scaling = SCALE_MINMAX,distance = DIST_MANHATTAN) #Distance - Manhattan, Scaling = MinMax
    main(scaling = SCALE_ZNORMAL) #Distance - Euclidean, Scaling - Z-score normalization
    main(scaling = SCALE_ZNORMAL,distance = DIST_MANHATTAN) #Distance - Manhattan, Scaling - Z-score normalization