import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os



    

class Logistic_Regression:
    def __init__(self,learning_rate = 0.01,n_itrs=1000):
        self.learning_rate= learning_rate
        self.n_itrs  = n_itrs
        self.weight = None
        self.bias = None

    def fit(self,X_train, Y_train):
        self.bias = 0
        self.n_samples , self.n_features = X_train.shape
        self.weight= np.zeros(self.n_features)

        for iter in range(self.n_itrs):
            Y_pred = self.sigmoid(X_train)
            
            dw =(1/self.n_samples) * np.dot(X_train.T,(Y_pred-Y_train)) # we mutiply the x transpose by the error value 
            db = (1/ self.n_samples)* np.sum(Y_pred-Y_train)
    
            self.weight -= self.learning_rate *dw
            self.bias -= self.learning_rate *db


    def predict(self,X_test):
        y_pred = self.sigmoid(X_test)
        return [0 if y<=0.5 else 1 for y in y_pred]
        
        
    def sigmoid(self,X): #sigmoid fn
        z = self.calc_power(X)
        z = np.clip(z, -500, 500)
        return 1/(1+ np.exp(-z))
                  
    def calc_power(self,X_test):
        return np.dot(X_test.values, self.weight) + self.bias

    def accuracy(self,Y_test,Y_predict):

        return np.sum(Y_test==Y_predict)/len(Y_test)



    





if __name__ =='__main__':


    #load data 


    import kagglehub
    
    # Download latest version
    path = 'D:/Downloads/archive'
     #Data processing
    
    print("Path to dataset files:", path)
    df = pd.read_csv(path+ '/train.csv')
    # fill missing values
    age_mean = df['Age'].mean()
    df['Age'] = df['Age'].fillna( age_mean )

    Cabin_mode = df['Cabin'].mode()[0]
    df['Cabin'] = df['Cabin'].fillna(Cabin_mode)

    #encoding
    encoder = LabelEncoder()
    encoder.fit(df['Sex'])
    df['Sex'] = encoder.transform(df['Sex'])
    
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_dummies], axis=1)

    # Now drop the original text column
    df.drop('Embarked', axis=1, inplace=True)
    
    # matplot & Histograms
    plt.figure(1)
    plt.hist(df['Age'])
    plt.title("Age before Scalling")
    
    plt.figure(2)
    plt.hist(df['Fare'])
    plt.title("Fare before Scalling")

    #scaling  
    age_range = np.max(df['Age'])- np.min(df['Age'])
    df['Age'] = (df['Age']- np.min(df['Age']))/age_range

    fare_mean = np.mean(df['Fare'])
    df['Fare'] = (df['Fare']- fare_mean)/np.std(df['Fare'])
    plt.figure(3)
    plt.hist(df['Age'])
    plt.title("Age after Scalling")
    
    plt.figure(4)
    plt.hist(df['Fare'])
    plt.title("Fare after Scalling")
    plt.show()

    # np.shuffle(df)
    Y= df['Survived']
    # List of columns to remove from our "clues"
    columns_to_drop = [
        'PassengerId',  # Just a random ID number
        'Name',         # Text, not useful for math
        'Ticket',       # Text, not useful
        'Cabin',        # Text, not useful
        'Survived'      # This is the ANSWER, we can't use it as a clue!
    ]
    
    # Create X by dropping all those columns
    X = df.drop(columns=columns_to_drop)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,shuffle = True)

    
    # min max scalling of columns age 
    #label

    # 
    #split

    clsf = Logistic_Regression()
    clsf.fit(X_train.values, Y_train)
    predictions = clsf.predict(X_test)
    
    # 4. Calculate and print accuracy
    acc = clsf.accuracy(Y_test, predictions)
    print(f"Model Accuracy: {acc}")
    
    


    
    
