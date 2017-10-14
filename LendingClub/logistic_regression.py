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
p = loansdata.boxplot('Interest.Rate','FICO.Range')
q = p.set_xticklabels(['640','','','','660','','','','680','','','','700', '720','','','','740','','','','760','','','','780','','','','800','','','','820','','','','840'])
q0 = p.set_xlabel('FICO Score')
q1 = p.set_ylabel('Interest Rate %')
q2 = p.set_title('Lending Rate Plot')

#Create a new data frame with selected columns for analysing data
loansmin = loansdata.filter(['Interest.Rate','FICO.Score','Loan.Length','Monthly.Income','Amount.Requested'], axis = 1)

# Do logistic regression model for the question What is the probability of getting a Loan, from the Lending Club, 
# of 10,000 dollars at 12 per cent or less with a FICO Score of 720?
# print loansmin.head()

loansmin['TF']=loansmin['Interest.Rate']<=0.12
# print loansmin.head()

# statsmodels requires us to add a constant column representing the intercept
loansmin['Intercept']=1.0

# identify the independent variables 
ind_cols=['FICO.Score', 'Amount.Requested', 'Intercept']
logit = sm.Logit(loansmin['TF'], loansmin[ind_cols])
#Do linear regression
result=logit.fit()

# get the fitted coefficients from the results
coefficient = result.params
# print coefficient
print("Trying multiple FICO Loan Amount combinations: ")
print('----')
print("fico=720, amt=10,000")
print(pz(720,10000,coefficient))
print("fico=720, amt=20,000")
print(pz(720,20000,coefficient))
print("fico=720, amt=30,000")
print(pz(720,30000,coefficient))
print("fico=820, amt=10,000")
print(pz(820,10000,coefficient))
print("fico=820, amt=20,000")
print(pz(820,20000,coefficient))
print("fico=820, amt=30,000")
print(pz(820,30000,coefficient))
print("fico=820, amt=63,000")
print(pz(820,63000,coefficient))
print("fico=820, amt=50,000")
print(pz(820,50000,coefficient))

def pz(fico_score, amount, coefficient):
	# compute  linear expression by multipyling the inputs by their respective coefficients.
	# the coefficient array has the intercept coefficient at the end
	z = coefficient[0] * fico_score + coefficient[1] * amount + coefficient[2]
	return 1 / (1 + np.exp(-1 * z))
