# S05-005

# Compute the necessary sample values
n = 7
s = np.sqrt(62)

# Critical point computation with the Student's t:
cl = 0.95
alpha = 1 - cl
crit_point_left = stats.chi2.isf(alpha/2, df = n - 1)
crit_point_right = stats.chi2.ppf(alpha/2, df = n - 1)

# Formula for the confidence interval
conf_int = (n - 1) * s**2 / np.array([crit_point_left, crit_point_right])
print("The confidence interval is [{:.4}, {:.4}]".format(conf_int[0], conf_int[1]))