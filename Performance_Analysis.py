'''
DATASET - Took "income_evaluation" dataset as sample

OBJECTIVE - Compare the performance of three models namely Decision Tree, K-NN, and
Bayesian classifier over it.

OBSERVATION - Try putting differnt values of k. As the value of k is increased, the accuracy
also increases but its difficult to choose an appropraite value of k

CONCLUSION - for the dataset choosen Bayesian Classifier is the best model as the accuracy
is the highest and k nearest neighbor is the worst as it requires a lot of computation and
hence have lowest accuracy.
We cannot say best model in general as model should be choosen according to the dataset
available as accuracy of model vary accoring to dataset and its dimensions.
'''

#Importing pandas
import pandas as pd

#Importing numpy 
import numpy as np

#Importing LabelEncoder for encoding
from sklearn.preprocessing import LabelEncoder

#Importing train_test_split for train test splitting
from sklearn.model_selection import train_test_split

#Importing DecisionTreeClassifier for decision tree object
from sklearn.tree import DecisionTreeClassifier

#Importing GaussianNB for Bayesian object
from sklearn.naive_bayes import GaussianNB

#Importing KNeighborsClassifier for K nearest neighbor object
from sklearn.neighbors import KNeighborsClassifier

#for checking testing results
from sklearn.metrics import classification_report, accuracy_score

def read_data(fileName):
    
    '''
    Assuming that file is of csv type.
    Purpose of the function is to take a csv file name/path from the user and return the
    data frame.
    Input: fileName - name of the csv file to read
    Output: return the data frame obtained
    '''
    
    #reading the data into data frame format
    df = pd.read_csv(fileName)
    
    #returning the data frame obtained
    return df

def preprocess(df):

    '''
    Purpose of the function is to preprocess the data and separate the data into 2 parts -
    one - the attribute to be predicted and second - remaining data and return it.
    Input: df - data frame obtained after reading the csv file
    Output: return x - series containing data excluding the data for attribute to be predicted
                   y - series containing data for attribute to be predicted
    '''

    #The data set contains '?' in each columns for missing values.
    #Traversing through the columns of data frame and replaing '?' with NaN    
    for column in df.columns:
        df[column].replace(' ?', np.NaN, inplace=True)

    #Droping rows from the data frame and reseting index
    df = df.dropna().reset_index().drop(columns=['index'])

    #Preprocess the columns in data frame as all are not of integer or float type.
    for cols in df.columns:
        le = LabelEncoder()
        df[cols]=le.fit_transform(df[cols])

    #droping the income column from the dataset as income is the attribute to be predicted and
    #saving it in x.
    x = df.drop(' income', axis = 1)

    #saving data of income column in y
    y = df[' income']
    
    #returning x and y
    return x,y
    
def check_accuracy(x,y,k):
    
    '''
    Purpose of the function is to take different instances for training and test data, and run
    each model on these instances and return the average accuracy of each model.
    Input: x - series containing data excluding the data for attribute to be predicted
           y - series containing data for attribute to be predicted
           k - no. of nearest neigbhors
    Output: return a series containing average accuracy of 3 models
    '''

    #Dictionary of model with model name as key and function as values
    modelslist = {"Decision Tree":DecisionTreeClassifier(),"K Neigbhor Classifier":KNeighborsClassifier(k),"Bayesian classifier":GaussianNB()}

    #list to store accuracy for all models in differnt samples
    acc_list=[]
    for i in range(20):
        #list to store accuracy for one sample
        acc_sample=[]
        
        #using train_test_split() to segregate the data into test and training data
        x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x, y, test_size = 0.3)

        #uncomment code below if you want to see the training and testing data 
        '''
        print("Sample",i+1,"\n")
        print("\nx training data\n",x_train_data.values)
        print("\nx test data\n",x_test_data.values)
        print("\ny training data\n",y_train_data)
        print("\ny test data\n",y_test_data)
        '''
        
        #traversing the dictionary and and training the model with training data and testing
        #model on test data
        for model in modelslist:
            modelslist[model].fit(x_train_data.values, y_train_data)
            predictions =modelslist[model].predict(x_test_data.values)
            #uncomment the code below if you want to see classification report for differnt model
            '''
            print("\nClassification report for ",model," is :\n",classification_report(y_test_data,predictions))
            '''
            #calculating the accuracy_score and appending in acc_sample list
            acc_sample.append(accuracy_score(predictions,y_test_data)*100)
        
        #appending accuracy for each sample as a list in acc_list list 
        acc_list.append(acc_sample)

    #converting list to dataframe
    avg_acc=pd.DataFrame(acc_list)
    
    #calculating mean accuracy for each model
    avg_acc=avg_acc.mean()
    
    #returning mean accuracy
    return avg_acc

def main():
    '''
    Purpose of this function is to invoke other functions and display the result.
    Input: none
    Output: return nothing, just displaying the mean accuracy of 3 models
    '''
    #file name
    file='income_evaluation.csv'
    
    #invoking read_data() to read csv file
    data=read_data(file)
    
    #invoking the preprocess() to preprocess the data
    x,y=preprocess(data)
    
    #taking no. of neigbhors from user & invoking check_accuracy() to train and test the model
    k=int(input("Enter the no. of neighbors needed : "))
    accDecision,accKneighbor,accBayesian=check_accuracy(x,y,k)
    
    #displaying the mean accuracy of all 3 models
    print("Accuracy of Decision tree model is : ","{:.2f}".format(accDecision),"%")
    print("Accuracy of K nearest neighbor model is : ","{:.2f}".format(accKneighbor),"%")
    print("Accuracy of Bayesian Classifier model is : ","{:.2f}".format(accBayesian),"%")

if __name__=='__main__':
    main()
