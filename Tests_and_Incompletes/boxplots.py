import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# The arrays
hs_dc_gowt = [0.21893582130394748, 0.18907499928974103,0.17858939512599623,0.19213473222121372, 0.2511605825102435,0.29163760946180595]
dice_mh_GOWT1 = [0.8479139784946237, 0.7897543357524959, 0.7991618631153515, 0.7936914767322101, 0.8372021654002431, 0.83254629515168]
dice_scores = np.array([hs_dc_gowt, dice_mh_GOWT1])
x = ['H', 'MH']
# Seaborn requires a dataframe
dataframe = pd.DataFrame(data = np.transpose(dice_scores), columns=x)
print(dataframe)
ax = sns.swarmplot(data = dataframe)
plt.show()