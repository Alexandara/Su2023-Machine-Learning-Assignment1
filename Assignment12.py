import Assignment11
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor

def linearRegressionModel(testsize, learnrate, iterations):
    '''
        Part 3: Training and Test Data
        Although SciKit Learn has linear regression libraries, we're just using it here
        to divide the data into 80% training data and 20% testing data.
        '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=5)
    '''
        Part 4: Train a linear regression model
        We train our linear regression model using code from the June 1st lecture
        in class, modified for our use case. 
    '''
    model = SGDRegressor(random_state=42, max_iter=iterations, tol=1e-3,
                         penalty=None, learning_rate=learnrate)
    model.fit(X_train, y_train)
    '''
        Part 5: Test the linear regression model
    '''
    y_pred = model.predict(X_test)
    rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
    '''
        Part 6: Document attempts and discern quality
    '''
    f = open("a12logs.txt", "a")
    f.write("Regression Attempted at " + str(datetime.datetime.now()))
    f.write("\nTrain to Test Ratio: " + str(1 - testsize) + "/" + str(testsize))
    f.write("\nLearn Rate: " + str(learnrate))
    f.write("\nIterations: " + str(iterations))
    f.write("\nMy Error: " + str(rmse))
    f.close()
    return rmse

if __name__ == '__main__':
    '''
    Dataset: Computer Hardware
    Attribute Information:
       1. vendor name: 30 
          (adviser, amdahl,apollo, basf, bti, burroughs, c.r.d, cambex, cdc, dec, 
           dg, formation, four-phase, gould, honeywell, hp, ibm, ipl, magnuson, 
           microdata, nas, ncr, nixdorf, perkin-elmer, prime, siemens, sperry, 
           sratus, wang)
       2. Model Name: many unique symbols
       3. MYCT: machine cycle time in nanoseconds (integer)
       4. MMIN: minimum main memory in kilobytes (integer)
       5. MMAX: maximum main memory in kilobytes (integer)
       6. CACH: cache memory in kilobytes (integer)
       7. CHMIN: minimum channels in units (integer)
       8. CHMAX: maximum channels in units (integer)
       9. PRP: published relative performance (integer)
      10. ERP: estimated relative performance from the original article (integer) 
    '''
    path = 'https://personal.utdallas.edu/~art150530/machine.data'
    '''
    Part 2: Pre-processing 
    We create a dataframe from the data that includes all columns except for the 
    name of the vendor, the model name, the published relative performance (which will be
    our label) and the estimated relative performance, which is a prediction by
    some other machine learning researchers and would thus be a little unfair for
    us to use. 
    '''
    df = pd.read_csv(path, skiprows=1, index_col=False,
                     names=['NAME', 'MODEL', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP'],
                     usecols=['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP'])
    answers = pd.read_csv(path, skiprows=1, index_col=False,
                          names=['NAME', 'MODEL', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP'],
                          usecols=['PRP', 'ERP'])
    # Drop all rows with NA values
    df = df.dropna(axis=0, how='any')
    # Drop all rows that are duplicates
    df = df.drop_duplicates()
    # PRP is our label
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    testsize = 0.8
    testsizeArr = [.2, .3, .4, .5, .6, .7, .8, .9]
    learnrate = 'adaptive'
    iterations = 100
    iterationsArr = [100, 150, 200, 300, 500, 1000, 1500, 5000]
    testSizeError = []
    iterationsError = []
    f = open("a12logs.txt", "a")
    f.write("AUTOMATIC TESTING START\n")
    f.close()
    for i in testsizeArr:
        testSizeError.append(linearRegressionModel(i, learnrate, iterations))
    Assignment11.plot(np.array(testsizeArr), np.array(testSizeError), "Test Size Variations (scikit)", "Test Size", "Mean Squared Error")
    for i in iterationsArr:
        iterationsError.append(linearRegressionModel(testsize, learnrate, i))
    Assignment11.plot(np.array(iterationsArr), np.array(iterationsError), "Iteration Variations (scikit)", "Iterations", "Mean Squared Error")
    print(linearRegressionModel(testsize, learnrate, iterations))