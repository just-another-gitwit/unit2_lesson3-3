import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# load the reduced version of the Lending Club Dataset
loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

# using lambda...
# remove % from Interest.Rate
# convert Interest.Rate from string to float
# divide float by 100 to convert from percent
# round outcome to 4 digits
cleanInterestRate = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))

print cleanInterestRate[0:5]

# using lambda...
# remove months from Loan.Length
# convert Loan.Length from string to integer
cleanLoanLength = loansData['Loan.Length'].map(lambda x: int(x.rstrip(' months')))

print cleanLoanLength[0:5]

# using list comprehension...
# split FICO.Range and turn into string
# turn FICO.Range into an integer
# keep lower number and save into new column FICO.Score
loansData['FICO.Score'] = [int(val.split('-')[0]) for val in loansData['FICO.Range']]

cleanFICOScore = loansData['FICO.Score']

print cleanFICOScore[0:5]

# plot histogram of FICO scores
plt.figure()
p = loansData['FICO.Score'].hist()
plt.show()

# plot scatter matrix
plt.figure()
a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10))
plt.show()

# plot scatter matrix with histogram diagonal
plt.figure()
a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10), diagonal='hist')
plt.show()

# linear regression analysis

intrate = cleanInterestRate #loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = cleanFICOScore #loansData['FICO.Score']

# The dependent variable
y = np.matrix(intrate).transpose()
# The independent variables shaped as columns
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

x = np.column_stack([x1,x2])

X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

print f.summary()
