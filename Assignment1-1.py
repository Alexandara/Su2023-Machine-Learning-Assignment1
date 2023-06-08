import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import datetime

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
path = './machine.data'
'''
Part 2: Pre-processing 
We create a dataframe from the data that includes all columns except for the 
name of the vendor, the model name, the published relative performance (which will be
our label) and the estimated relative performance, which is a prediction by
some other machine learning researchers and would thus be a little unfair for
us to use. 
'''
df = pd.read_csv(path, skiprows=1, index_col=False,
                 names=['NAME','MODEL','MYCT', 'MMIN','MMAX','CACH', 'CHMIN','CHMAX','PRP','ERP'],
                 usecols=['MYCT', 'MMIN','MMAX','CACH', 'CHMIN','CHMAX', 'PRP'])
# Drop all rows with NA values
df = df.dropna(axis=0,how='any')
# Drop all rows that are duplicates
df = df.drop_duplicates()
# PRP is our label
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
'''
Part 3: Training and Test Data
Although SciKit Learn has linear regression libraries, we're just using it here
to divide the data into 80% training data and 20% testing data.
'''
testsize = .5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=5)
'''
Part 4: Train a linear regression model
We train our linear regression model using code from the gradient descent lab done 
in class, modified for our use case. 
'''
def ssr_gradient(x, w, lr):
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
def gradient_descent(
     gradient, X, y, start, learn_rate=0.1, n_iter=50, tolerance=1e-06
 ):
    vector = start
    tempdf = pd.concat([X, y], axis=1, join='inner')
    go = True
    for _ in range(n_iter):
        for index, row in tempdf.iterrows():
            vector, loss = gradient(row, vector, learn_rate)
            if np.all(np.abs(loss) <= tolerance):
                go = False
        if go == False:
            break
    print(vector)
    return vector

learnrate = 0.00000000000000000008
startweights = 0
iterations = 150
model = gradient_descent(
    ssr_gradient, X_train, y_train,
    start=np.array([startweights, startweights, startweights,
                    startweights, startweights, startweights, startweights],dtype='float128'),
    learn_rate=learnrate,
    n_iter=iterations)
'''
Part 5: Test the linear regression model
'''
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
    sum = (float(row['PRP']) - float(y_pred)) * (float(row['PRP']) - float(y_pred))
    num = num + 1
MSE = sum / num
'''
Part 6: Document attempts and discern quality
'''
f = open("a1logs.txt", "a")
f.write("Regression Attempted at " + str(datetime.datetime.now()))
f.write("\nTrain to Test Ratio: " + str(1-testsize) + "/" + str(testsize))
f.write("\nLearn Rate: " + str(learnrate))
f.write("\nStarting Weights (all): " + str(startweights))
f.write("\nIterations: " + str(iterations))
f.write("\nError: " + str(MSE) + "\n")
f.close()