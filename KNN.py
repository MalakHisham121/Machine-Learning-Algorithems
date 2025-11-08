import numpy as np
import pandas as pd
from collections import Counter
from sklearn import datasets, preprocessing, model_selection,neighbors,metrics

class KNN:
    def __init__(self,k= 5 ):
        self.k =k


    
    def Eculedian_distance(self,x,y):
        '''
        sum = 0
        for feature_id in len(x):
            diff = (x[feature_idx] - y[feature_idx])**2
            sum += diff
        '''

        return np.sqrt(np.sum((x-y)**2))
        



    def predict(self,X_train,Y_train,x_predict):
        # x_predict is list
        #get distance of all dataset with the new point
        ans =[]
        for x in x_predict :
            distances = []
            for i in range(len(X_train)) :
                distances.append((Y_train[i],self.Eculedian_distance(x, X_train[i])))
                
                              
            distances= sorted(distances,key=lambda item: item[1])[:self.k]
            
            label = Counter([items[0] for items in distances]).most_common(1)[0][0]

            ans.append(label)
        return ans 
                
    def accuracy(self,y_predict, y_test):
        return np.sum(y_predict == y_test)/len(y_predict)
        

if __name__ == '__main__':

    df = datasets.load_iris()
    X = df.data
    Y = df.target
    l = len(Y)
    shuffled = np.random.permutation(l)
    X= X[shuffled] 
    Y= Y[shuffled]
    print(X[:5])


    # feature scalling 
    #scaler = StandardScaler()
    #X= scaler.fit_transorm(X)
    # start loading the data
    #process it 
    # split it
    
    X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y, test_size =0.3)
    mean_x_train = np.mean(X_train,axis =0)
    #mean_x_test = np.mean(X_test)

    std_x_train = np.std(X_train,axis =0)
    #std_x_test = np.std(X_test)

    X_train= (X_train-mean_x_train)/std_x_train
    X_test = (X_test - mean_x_train)/ std_x_train

    
    
    algo = KNN( k=7 )
    
    algo.predict(X_train,Y_train,X_test)

    print ("acuuracy is : " ,algo.accuracy( algo.predict(X_train,Y_train,X_test), Y_test))
        


        
    
    
