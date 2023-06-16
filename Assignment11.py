import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
import matplotlib.pyplot as plt

class LinearRegressionModel:
    def __init__(self, testsize, learnrate, startweights, iterations):
        '''
        Part 3: Training and Test Data
        Although SciKit Learn has linear regression libraries, we're just using it here
        to divide the data into 80% training data and 20% testing data.
        '''
        self.testsize = testsize
        self.learnrate = learnrate
        self.startweights = startweights
        self.iterations = iterations
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.testsize, random_state=5)
        '''
        Part 4: Train a linear regression model
        We train our linear regression model using code from the gradient descent lab done 
        in class, modified for our use case. 
        '''
        model = self.gradient_descent(
            self.ssr_gradient, X_train, y_train,
                            start=np.array([self.startweights, self.startweights, self.startweights,
                            self.startweights, self.startweights, self.startweights, self.startweights],dtype='float128'))
        '''
        Part 5: Test the linear regression model
        '''
        combineddftrain = pd.concat([X_train, y_train], axis=1, join='inner')
        sum = 0
        num = 0
        for index, row in combineddftrain.iterrows():
            y_pred = model[0] + model[1] * row['MYCT'] \
                     + model[2] * row['MMIN'] \
                     + model[3] * row['MMAX'] \
                     + model[4] * row['CACH'] \
                     + model[5] * row['CHMIN'] \
                     + model[6] * row['CHMAX']
            sum = sum + (float(row['PRP']) - float(y_pred)) * (float(row['PRP']) - float(y_pred))
            num = num + 1
        self.myTrainMSE = sum / num
        combineddf = pd.concat([X_test, y_test], axis=1, join='inner')
        sum = 0
        num = 0
        for index, row in combineddf.iterrows():
            y_pred = model[0] + model[1] * row['MYCT'] \
                  + model[2] * row['MMIN'] \
                  + model[3] * row['MMAX'] \
                  + model[4] * row['CACH'] \
                  + model[5] * row['CHMIN'] \
                  + model[6] * row['CHMAX']
            sum = sum + (float(row['PRP']) - float(y_pred)) * (float(row['PRP']) - float(y_pred))
            num = num + 1
        self.myTestMSE = sum / num
        sum = 0
        num = 0
        for index, row in answers.iterrows():
            sum = sum + (float(row['PRP']) - float(row['ERP'])) * (float(row['PRP']) - float(row['ERP']))
            num = num + 1
        self.theirMSE = sum / num
        '''
        Part 6: Document attempts and discern quality
        '''
        f = open("a11logs.txt", "a")
        f.write("Regression Attempted at " + str(datetime.datetime.now()))
        f.write("\nTrain to Test Ratio: " + str(1-self.testsize) + "/" + str(self.testsize))
        f.write("\nLearn Rate: " + str(self.learnrate))
        f.write("\nStarting Weights (all): " + str(self.startweights))
        f.write("\nIterations: " + str(self.iterations))
        f.write("\nMy Training Error: " + str(self.myTrainMSE))
        f.write("\nMy Test Error: " + str(self.myTestMSE))
        f.write("\nOriginal Researcher's Error: " + str(self.theirMSE) + "\n")
        f.close()

    def get_results(self):
        return self.myTrainMSE, self.myTestMSE
    def gradient_descent(
         self, gradient, X, y, start, tolerance=1e-06
     ):
        vector = start
        tempdf = pd.concat([X, y], axis=1, join='inner')
        go = True
        for _ in range(self.iterations):
            for index, row in tempdf.iterrows():
                vector, loss = gradient(row, vector, self.learnrate)
                if np.all(np.abs(loss) <= tolerance):
                    go = False
            if go == False:
                break
        return vector

    def ssr_gradient(self, x, w, lr):
        y_pred = w[0] + w[1] * x['MYCT'] \
              + w[2] * x['MMIN'] \
              + w[3] * x['MMAX'] \
              + w[4] * x['CACH'] \
              + w[5] * x['CHMIN'] \
              + w[6] * x['CHMAX']
        loss = x['PRP'] - y_pred
        ly = 2 * loss
        yw0 = 1
        yw1 = x['MYCT']
        yw2 = x['MMIN']
        yw3 = x['MMAX']
        yw4 = x['CACH']
        yw5 = x['CHMIN']
        yw6 = x['CHMAX']
        w0up = w[0] - lr * (ly * yw0)
        w1up = w[1] - lr * (ly * yw1)
        w2up = w[2] - lr * (ly * yw2)
        w3up = w[3] - lr * (ly * yw3)
        w4up = w[4] - lr * (ly * yw4)
        w5up = w[5] - lr * (ly * yw5)
        w6up = w[6] - lr * (ly * yw6)
        return np.array([w0up, w1up, w2up, w3up, w4up, w5up, w6up], dtype='float128'), loss

def plot(X,Y,name,xlabel,ylabel):
    plt.plot(X,Y,linestyle="",marker="o")
    plt.title(name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.autoscale()
    plt.savefig(name + ".png")
    plt.clf()

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

    testsize = 0.4
    testsizeArr = [.2,.3,.4,.5,.6,.7,.8,.9]
    learnrate = 0.00000000000000000008
    learnrateArr = [0.000000000000008,
                    0.00000000000000008, 0.0000000000000000008,
                    0.000000000000000000008, 0.00000000000000000000008,
                    0.0000000000000000000000008, 0.000000000000000000000000008]
    startweights = 0
    startweightsArr = [-.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5]
    iterations = 100
    iterationsArr = [100,150,200,300,500,1000,1500,5000]
    testSizeError = []
    learnrateError = []
    startweightsError = []
    iterationsError = []
    ErrorTrain = []
    ErrorTest = []
    currbest = [0, 0]
    f = open("a11logs.txt", "a")
    f.write("AUTOMATIC TESTING START\n")
    f.close()
    for i in testsizeArr:
        train, test = LinearRegressionModel(i, learnrate, startweights, iterations).get_results()
        if test > currbest[0] or currbest[0] == 0:
            currbest = [test, i]
        testSizeError.append(test)
        ErrorTrain.append(train)
        ErrorTest.append(test)
    testsizeBest = currbest[1]
    currbest = [0, 0]
    plot(np.array(testsizeArr), np.array(testSizeError), "Test Size Variations", "Test Size", "Mean Squared Error")
    for i in learnrateArr:
        train, test = LinearRegressionModel(testsize, i, startweights, iterations).get_results()
        learnrateError.append(test)
        ErrorTrain.append(train)
        ErrorTest.append(test)
        if test > currbest[0] or currbest[0] == 0:
            currbest = [test, i]
    learnrateBest = currbest[1]
    currbest = [0, 0]
    plot(np.array(learnrateArr), np.array(learnrateError), "Learning Rate Variations", "Learning Rate", "Mean Squared Error")
    for i in startweightsArr:
        train, test = LinearRegressionModel(testsize, learnrate, i, iterations).get_results()
        startweightsError.append(test)
        ErrorTrain.append(train)
        ErrorTest.append(test)
        if test > currbest[0] or currbest[0] == 0:
            currbest = [test, i]
    startweightsBest = currbest[1]
    currbest = [0, 0]
    plot(np.array(startweightsArr), np.array(startweightsError), "Starting Weight Variations", "Starting Weights (all)", "Mean Squared Error")
    for i in iterationsArr:
        train, test = LinearRegressionModel(testsize, learnrate, startweights, i).get_results()
        iterationsError.append(test)
        ErrorTrain.append(train)
        ErrorTest.append(test)
        if test > currbest[0] or currbest[0] == 0:
            currbest = [test, i]
    iterationsBest = currbest[1]
    currbest = [0, 0]
    plot(np.array(iterationsArr), np.array(iterationsError), "Iteration Variations", "Iterations", "Mean Squared Error")
    plot(np.array(ErrorTrain), np.array(ErrorTest), "Train Error Vs. Test Error", "Train Error", "Test Error")
    print(LinearRegressionModel(testsizeBest, learnrateBest, startweightsBest, iterationsBest).get_results())
