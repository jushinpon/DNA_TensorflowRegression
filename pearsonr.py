
from scipy.stats import pearsonr
 
x = [0.5, 0.4, 0.6, 0.3, 0.6, 0.2, 0.7, 0.5]
y = [0.6, 0.4, 0.4, 0.3, 0.7, 0.2, 0.5, 0.6]
print(type(x))
print(pearsonr(x, y))
