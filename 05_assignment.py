'''
Assignment #5
1. Add / modify code ONLY between the marked areas (i.e. "Place code below")
2. Run the associated test harness for a basic check on completeness. A successful run of the test cases does not 
    guarantee accuracy or fulfillment of the requirements. Please do not submit your work if test cases fail.
3. To run unit tests simply use the below command after filling in all of the code:
    python 07_assignment.py
  
4. Unless explicitly stated, please do not import any additional libraries but feel free to use built-in Python packages
5. Submissions must be a Python file and not a notebook file (i.e *.ipynb)
6. Do not use global variables unless stated to do so
7. Make sure your work is committed to your master branch in Github
Packages required:
pip install cloudpickle==0.5.6 (this is an optional install to help remove a deprecation warning message from sklearn)
pip install sklearn
'''
# core
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ml
from sklearn import datasets as ds
from sklearn import linear_model as lm
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split as tts

# infra
import unittest

# ------ Place code below here \/ \/ \/ ------
# import plotly library and enter credential info here

import chart_studio
import chart_studio.plotly as py
import plotly.graph_objects as go

chart_studio.tools.set_credentials_file(username='lawsvin', api_key='sqZXF2FGrQYAoEgvsvwy')

# ------ Place code above here /\ /\ /\ ------





# ------ Place code below here \/ \/ \/ ------
# Load datasets here once and assign to variables iris and boston

iris = ds.load_iris()
boston = ds.load_boston()

# ------ Place code above here /\ /\ /\ ------




# 10 points
def exercise01():
    '''
        Data set: Iris
        Return the first 5 rows of the data including the feature names as column headings in a DataFrame and a
        separate Python list containing target names
    '''

    # ------ Place code below here \/ \/ \/ ------
    
    # Return the first 5 rows of the data including the feature names as column headings.
    df_iris = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    df_first_five_rows = df_iris[0:5]

    # Python list containing target names.
    target_names = list(iris['target_names'])
    
    # ------ Place code above here /\ /\ /\ ------

    return df_first_five_rows, target_names

# 15 points
def exercise02(new_observations):
    '''
        Data set: Iris
        Fit the Iris dataset into a kNN model with neighbors=5 and predict the category of observations passed in 
        argument new_observations. Return back the target names of each prediction (and not their encoded values,
        i.e. return setosa instead of 0).
    '''

    # ------ Place code below here \/ \/ \/ ------

    # Redefine the iris dataframe here.
    df_iris = pd.DataFrame(iris['data'], columns=iris['feature_names'])

    # Create a list of species.
    species=[iris['target_names'][i] for i in iris['target']]

    # Add the species list to the iris dataframe.
    df_iris['species']=species

    # Create arrays for the features and the response variable.
    y = df_iris['species']
    X = df_iris.drop('species', axis=1).values

    # Create a k-NN classifier with 5 neighbors.
    knn = KNN(n_neighbors=5)

    # Fit the classifier to the training data.
    knn.fit(X,y)

    # Predict the category for the training data X.
    knn.predict(X)

    # Predict the category for the new observations.
    iris_predictions = knn.predict(new_observations)

    # ------ Place code above here /\ /\ /\ ------

    return iris_predictions

# 15 points
def exercise03(neighbors,split):
    '''
        Data set: Iris
        Split the Iris dataset into a train / test model with the split ratio between the two established by 
        the function parameter split.
        Fit KNN with the training data with number of neighbors equal to the function parameter neighbors
        Generate and return back an accuracy score using the test data was split out
    '''
    random_state = 21

    
    # ------ Place code below here \/ \/ \/ ------

    # Redefine the iris dataframe and add the species column to the dataframe.
    df_iris = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    species=[iris['target_names'][i] for i in iris['target']]
    df_iris['species']=species

    # Create arrays for the features and the response variable.
    y = df_iris['species']
    X = df_iris.drop('species', axis=1).values

    # Split into training and test set.
    X_train, X_test, y_train, y_test = tts(X, y, test_size = split, random_state=random_state, stratify=y)

    # Create a k-NN classifier with passed in argument neighbors.
    knn = KNN(n_neighbors=neighbors)

    # Fit the classifier to the training data.
    knn.fit(X_train, y_train)

    # Generate an accuracy score using the test data.
    knn_score = knn.score(X_test, y_test)

    # ------ Place code above here /\ /\ /\ ------

    return knn_score

# 20 points
def exercise04():
    '''
        Data set: Iris
        Generate an overfitting / underfitting curve of kNN each of the testing and training accuracy performance scores series
        for a range of neighbor (k) values from 1 to 30 and plot the curves (number of neighbors is x-axis, performance score 
        is y-axis on the chart). Return back the plotly url.
    '''
    
    # ------ Place code below here \/ \/ \/ ------

    # Redefine the iris dataframe and add the species column to the dataframe.
    df_iris = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    species=[iris['target_names'][i] for i in iris['target']]
    df_iris['species']=species

    # Create arrays for the features and the response variable.
    y = df_iris['species']
    X = df_iris.drop('species', axis=1).values

    # Split into training and test set (lets the test_size and random_state parameter options to be default as 
    # they're not mentioned in the requirement of this exercise)
    X_train, X_test, y_train, y_test = tts(X, y, stratify=y)

    # Setup arrays to store training and testing accuracy performance scores series.
    neighbors = np.arange(1, 30)
    training_accuracy = np.empty(len(neighbors))
    testing_accuracy = np.empty(len(neighbors))

    # Loop over different values of neighbors, k.
    for i, k in enumerate(neighbors):
        # Setup a k-NN Classifier with k neighbors.
        knn = KNN(n_neighbors=k)

        # Fit the classifier to the training data.
        knn.fit(X_train, y_train)
    
        # Compute accuracy on the training set.
        training_accuracy[i] = knn.score(X_train, y_train)

        # Compute accuracy on the testing set.
        testing_accuracy[i] = knn.score(X_test, y_test)

    # Generate a scatter plotly URL for both training and testing accuracy versus varying number of neighbors. 
    Trace1 = go.Scatter(x=neighbors, y=training_accuracy, name='Training Accuracy')
    Trace2 = go.Scatter(x=neighbors, y=testing_accuracy, name='Testing Accuracy')
    data = [Trace1, Trace2]

    layout = go.Layout(title=go.layout.Title(text='k-NN: Varying Number of Neighbors',xref='paper',x=0),
    xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text='Number of Neighbors',font=dict(family='Courier New, monospace',size=18,color='#7f7f7f'))),
    yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text='Accuracy',font=dict(family='Courier New, monospace',size=18,color='#7f7f7f'))))

    figure = go.Figure(data=data, layout=layout)
    plotly_overfit_underfit_curve_url = py.plot(figure, filename = 'basic-line', auto_open=True)

    # ------ Place code above here /\ /\ /\ ------

    return plotly_overfit_underfit_curve_url

# 10 points
def exercise05():
    '''
        Data set: Boston
        Load sklearn's Boston data into a DataFrame (only the data and feature_name as column names)
        Load sklearn's Boston target values into a separate DataFrame
        Return back the average of AGE, average of the target (median value of homes or MEDV), and the target as NumPy values 
    '''

    # ------ Place code below here \/ \/ \/ ------

    # Turn the boston data into dataframe with the feature names as column names.
    df_boston_data = pd.DataFrame(boston['data'], columns=boston['feature_names'])

    # Turn the boston target values into a dataframe. 
    df_boston_target = pd.DataFrame(boston['target'])

    # Get the average of age.
    average_age = df_boston_data['AGE'].mean()

    # Get the average of target values.
    average_medv =df_boston_target[0].mean()

    # Get the target values as Numpy values.
    medv_as_numpy_values = np.array(df_boston_target[0])
   
    # ------ Place code above here /\ /\ /\ ------

    return average_age, average_medv, medv_as_numpy_values

# 10 points
def exercise06():
    '''
        Data set: Boston
        In the Boston dataset, the feature PTRATIO refers to pupil teacher ratio.
        Using a matplotlib scatter plot, plot MEDV median value of homes as y-axis and PTRATIO as x-axis
        Return back PTRATIO as a NumPy array
    '''

    # ------ Place code below here \/ \/ \/ ------

    # Turn the boston data into dataframe with the feature names as column names.
    df_boston_data = pd.DataFrame(boston['data'], columns=boston['feature_names'])

    # Add the boston target values into df_boston_data dataframe. 
    df_boston_data['MEDV'] = boston['target']

    # Generate a scatter plot of median value of home versus pupil teacher ratio.
    plt.scatter(df_boston_data['PTRATIO'], df_boston_data['MEDV'])
    plt.title('Median Value of Home vs Pupil Teacher Ratio')
    plt.legend()
    plt.xlabel('PTRATIO')
    plt.ylabel('Price')

    # Get the PTRATIO feature as a numpy array.
    X_ptratio = np.array(df_boston_data['PTRATIO'])

    # ------ Place code above here /\ /\ /\ ------

    return X_ptratio

# 20 points
def exercise07():
    '''
        Data set: Boston
        Create a regression model for MEDV / PTRATIO and display a chart showing the regression line using matplotlib
        with a backdrop of a scatter plot of MEDV and PTRATIO from exercise06
        Use np.linspace() to generate prediction X values from min to max PTRATIO
        Return back the regression prediction space and regression predicted values
        Make sure to labels axes appropriately
    '''

    # ------ Place code below here \/ \/ \/ ------

    # Turn the boston data into dataframe with the feature names as column names.
    df_boston_data = pd.DataFrame(boston['data'], columns=boston['feature_names'])

    # Add the boston target values into df_boston_data dataframe. 
    df_boston_data['MEDV'] = boston['target']

    # Create arrays for features and target variable.
    y = df_boston_data['MEDV'].values
    X = df_boston_data['PTRATIO'].values

    # Reshape X and y
    y = y.reshape(-1,1)
    X = X.reshape(-1,1)

    # Create the regressor.
    reg = lm.LinearRegression()

    # Create the prediction space.
    prediction_space = np.linspace(min(X), max(X)).reshape(-1,1)

    # Fit the model to the data.
    reg.fit(X, y)

    # Compute predictions over the prediction space.
    reg_model = reg.predict(prediction_space)

    # Overlay the scatter plot from exercise06 with a linear regression line.
    plt.plot(prediction_space, reg_model, color='black', linewidth=3)
    plt.title('Median Value of Home vs Pupil Teacher Ratio \n with a Linear Regression Line')
    plt.show()

    # ------ Place code above here /\ /\ /\ ------

    return reg_model, prediction_space


class TestAssignment7(unittest.TestCase):
    def test_exercise07(self):
        rm, ps = exercise07()
        self.assertEqual(len(rm),50)
        self.assertEqual(len(ps),50)

    def test_exercise06(self):
        ptr = exercise06()
        self.assertTrue(len(ptr),506)

    def test_exercise05(self):
        aa, am, mnpy = exercise05()
        self.assertAlmostEqual(aa,68.57,2)
        self.assertAlmostEqual(am,22.53,2)
        self.assertTrue(len(mnpy),506)
        
    def test_exercise04(self):
         print('Skipping EX4 tests')     

    def test_exercise03(self):
        score = exercise03(8,.25)
        self.assertAlmostEqual(exercise03(8,.3),.955,2)
        self.assertAlmostEqual(exercise03(8,.25),.947,2)
    def test_exercise02(self):
        pred = exercise02([[6.7,3.1,5.6,2.4],[6.4,1.8,5.6,.2],[5.1,3.8,1.5,.3]])
        self.assertTrue('setosa' in pred)
        self.assertTrue('virginica' in pred)
        self.assertTrue('versicolor' in pred)
        self.assertEqual(len(pred),3)
    def test_exercise01(self):
        df, tn = exercise01()
        self.assertEqual(df.shape,(5,4))
        self.assertEqual(df.iloc[0,1],3.5)
        self.assertEqual(df.iloc[2,3],.2)
        self.assertTrue('setosa' in tn)
        self.assertEqual(len(tn),3)
     

if __name__ == '__main__':
    unittest.main()
