#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import statsmodels.api as sm

plt.figure()

#Read the data
loansdata = pd.read_csv("/home/users/Downloads/loansData.csv")

#Clean the data
#Remove text from loan duration
loansdata['Loan.Length'] = loansdata['Loan.Length'].replace(' months','',regex=True).astype(int)
# loansdata['Loan.Length']=loansdata['Loan.Length'].map(lambda x: x.rstrip(' months')).astype(int)

#Remove % sign from interest rate and Debt to Income Ratio fields
loansdata['Interest.Rate']=loansdata['Interest.Rate'].replace('%','',regex=True).astype('float')/100
# loansdata['Interest.Rate']=loansdata['Interest.Rate'].map(lambda x: x.rstrip('%')).astype('float')/100
loansdata['Debt.To.Income.Ratio']=loansdata['Debt.To.Income.Ratio'].replace('%','',regex=True).astype('float')/100
# loansdata['Debt.To.Income.Ratio']=loansdata['Debt.To.Income.Ratio'].map(lambda x: x.rstrip('%')).astype('float')/100

#Remove NA fields
loansdata = loansdata.dropna(axis=0, how='any')
columns=['Amount.Requested','Amount.Funded.By.Investors','Monthly.Income','Open.CREDIT.Lines','Revolving.CREDIT.Balance','Inquiries.in.the.Last.6.Months','Interest.Rate']
# p = np.percentile(loansdata[columns], 95, axis=0) # axis=0, compute over each of the seperate columns.

#Maintain starting point of FICO Score from range
#Split up FICO.Range
loansdata['FICO.Score'],loansdata['FICO.End']=zip(*loansdata['FICO.Range'].apply(lambda x: x.split('-',1)))

# This drops several columns at the same time.
loansdata.drop(loansdata[['FICO.End', 'FICO.Range']], axis=1, inplace=True)
#Convert FICO.Score to an integer
loansdata['FICO.Score'].astype(int)
# loansdata['FICO.Score']=loansdata['FICO.Range'].str.split('-').str[0].astype(int)

# loansdata['FICO.Score'] = pd.to_numeric(loansdata['FICO.Score'], errors='coerce')
fico = loansdata['FICO.Score']
p = fico.hist()

# loansdata['Interest.Rate'] = pd.to_numeric(loansdata['Interest.Rate'], errors='coerce')
p = loansdata.boxplot('Interest.Rate','FICO.Score')
q = p.set_xticklabels(['640','','','','660','','','','680','','','','700', '720','','','','740','','','','760','','','','780','','','','800','','','','820','','','','840'])
q0 = p.set_xlabel('FICO Score')
q1 = p.set_ylabel('Interest Rate %')
q2 = p.set_title('Lending Rate Plot')

#Create a new data frame with selected columns for analysing data
loansmin = loansdata.filter(['Interest.Rate','FICO.Score','Loan.Length','Monthly.Income','Amount.Requested'], axis = 1)

a = pd.scatter_matrix(loansmin,alpha=0.05,figsize=(10, 10), diagonal='hist')
# a = pd.scatter_matrix(loansmin,alpha=0.05,figsize=(10, 10), diagonal='kde')
# a = pd.scatter_matrix(loansmin,alpha=0.05,figsize=(8, 8), diagonal='kde')
# a = pd.scatter_matrix(loansmin,alpha=0.05,figsize=(12, 12), diagonal='kde')

interest_rate = loansmin['Interest.Rate']
loan_amount = loansmin['Amount.Requested']
fico_score = loansmin['FICO.Score']

y = np.matrix(interest_rate).transpose()
x1 = np.matrix(fico_score).transpose()
x2 = np.matrix(loan_amount).transpose()

x = np.column_stack([x1,x2])

X = sm.add_constant(x)

#Do Linear Regression model
model = sm.OLS(y,X)
f = model.fit()

print 'Coefficients: ', f.params[0:2]
# Coefficients:  [ 71.25697692  -0.08295292]
print 'Intercept: ', f.params[2]
# Intercept:  7.10252436947e-06
print 'P-Values: ', f.pvalues
# P-Values:  [ 0.          0.          0.00051312]
print 'R-Squared: ', f.rsquared
# R-Squared:  0.505386208114
