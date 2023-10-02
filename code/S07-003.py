# S07-003.py

# mpg  dataset

print("mpg data set \n")

# Read the data
mpg_link = "https://raw.githubusercontent.com/tidyverse/ggplot2/main/data-raw/mpg.csv"
mpg = pd.read_csv(mpg_link)

# Extract input and output variables
X = mpg.cty.values
X = X[:, np.newaxis]

Y = mpg.hwy.values

# Load the required function from the scikit module
from sklearn.linear_model import LinearRegression

# Create the model
modelXY = LinearRegression(fit_intercept=True)

# Fit the model to the data
XY_fit = modelXY.fit(X, Y)

# Print the slope and intercept
print("slope = ", XY_fit.coef_)
print("intercapt = ", XY_fit.intercept_)

# Print the coefficient of determination R^2
XY = np.column_stack((X[:, 0], Y))
print("\nR^2 = ", np.corrcoef(XY, rowvar=False))

# ----------------------------------------------
# Framingham data set (should already be in the data folder)
# ----------------------------------------------

print("-"*30)
print("Framingham data set")

framingham = pd.read_csv("./data/framingham.csv")

# Extract input and output variables
X = framingham.diaBP.values
X = X[:, np.newaxis]

Y = framingham.sysBP.values

# Load the required function from the scikit module
from sklearn.linear_model import LinearRegression

# Create the model
modelXY = LinearRegression(fit_intercept=True)

# Fit the model to the data
XY_fit = modelXY.fit(X, Y)

# Print the slope and intercept
print("slope = ", XY_fit.coef_)
print("intercapt = ", XY_fit.intercept_)

# Print the coefficient of determination R^2
XY = np.column_stack((X[:, 0], Y))
print("\nR^2 = ", np.corrcoef(XY, rowvar=False))
