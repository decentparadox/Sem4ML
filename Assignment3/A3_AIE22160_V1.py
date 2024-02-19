import numpy as np
import pandas as pd


file = pd.ExcelFile("Lab Session1 Data.xlsx")

df = pd.read_excel(file,sheet_name = "Purchase data")
df = df.dropna(axis=1)

print(df)

#finding the dimensionality of the vector space
print(str(df.shape))
#small mistake here as we got index column not required
#matrix_A = df.iloc[:,[1,2,3]]

#proper implementation
matrix_A = df[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']]
print("Matrix A \n",matrix_A)
matrix_C = df[['Payment (Rs)']]
print("Matrix C \n",matrix_C)

#Finding the rank of matrix
print("Rank of Matrix A = ",np.linalg.matrix_rank(matrix_A))
print("Rank of Matrix C =",np.linalg.matrix_rank(matrix_C))
matrix_inv_A = np.linalg.pinv(matrix_A)
print("Inverse of matrix A is \n",matrix_inv_A)
      
#finding the X matrix
matrix_X = np.dot(matrix_inv_A,matrix_C)
print(pd.DataFrame(matrix_X))

#let's find the rank of the matrix
print("Rank of Matrix A = ",np.linalg.matrix_rank(matrix_A))

#The matrix X is calculated by multiplying Ainverse and C
matrix_X = np.dot(matrix_inv_A,matrix_C)
print("The Matrix X which is our regression model in here is : ",pd.DataFrame(matrix_X))

#classifier model
market_survey = []
for i in range(10):
  matrix_D = matrix_A.loc[i]
  matrix_D = matrix_D[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']]
  result = np.dot(matrix_D,matrix_X)
  if result[0] >= 200:
    market_survey.append(1)
  else:
    market_survey.append(-1)
df['classification'] = market_survey

df

#Lab Session1 Data
import pandas as pd
import numpy as np
import statistics
import matplotlib
# matplotlib.use('TkAgg') // I need this line because i use arch linux
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file into a DataFrame
df = pd.read_excel('Lab Session1 Data.xlsx', sheet_name='IRCTC Stock Price')

# Calculate mean and variance of the Price data
price_mean = statistics.mean(df['Price'])
price_variance = statistics.variance(df['Price'])

# Filter data for Wednesdays
wednesday_data = df[df['Day'] == 'Wed']
wednesday_mean = statistics.mean(wednesday_data['Price'])

# Filter data for April
april_data = df[df['Month'] == 'Apr']
april_mean = statistics.mean(april_data['Price'])

# Calculate probability of making a loss
loss_probability = len(df[df['Chg%'] < 0]) / len(df)

# Calculate probability of making a profit on Wednesday
wednesday_profit_probability = len(wednesday_data[wednesday_data['Chg%'] > 0]) / len(wednesday_data)

# Calculate conditional probability of making profit given Wednesday
conditional_profit_probability = len(wednesday_data[wednesday_data['Chg%'] > 0]) / len(df[df['Day'] == 'Wed'])

# Make scatter plot of Chg% data against the day of the week
sns.scatterplot(x="Day", y="Chg%", data=df, hue="Day", palette="hls")

# Show results
print("Mean of Price data:", price_mean)
print("Variance of Price data:", price_variance)
print("Mean of Price data on Wednesdays:", wednesday_mean)
print("Mean of Price data in April:", april_mean)
print("Probability of making a loss:", loss_probability)
print("Probability of making a profit on Wednesday:", wednesday_profit_probability)
print("Conditional probability of making profit given Wednesday:", conditional_profit_probability)
plt.xlabel("Day")
plt.ylabel("Chg%")
plt.title("Scatter plot")
plt.show()