import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# The arrays
dc_gowt = [0.20095467802323116, 0.18615249210830676, 0.14723089179436713, 0.17247176784418858, 0.23942897882574002, 0.2800658197354881]
g_dc_gowt = [0.6049033767200361, 0.545156565294437, 0.607365210111527, 0.6337398735867533, 0.628395754259946, 0.6101785396524112]
m_dc_gowt = [0.6037089513424323, 0.5424736733429693, 0.5957841500444236, 0.6207683116695029, 0.6151315544639202, 0.6342945864207978]
hs_dc_gowt = [0.7305905136874669, 0.777013844515442, 0.7818812453546948, 0.7545458227027902, 0.7540217593453599, 0.7841191962191998]
gs_dc_gowt = [0.7891630367381934, 0.774270861172657, 0.7936481543066178, 0.778829001019368, 0.7923810256015992, 0.7848321671557089]
dice_mh_GOWT1 = [0.8479139784946237, 0.7897543357524959, 0.7991618631153515, 0.7936914767322101, 0.8372021654002431, 0.83254629515168]
dice_scores = np.array([dc_gowt, g_dc_gowt, m_dc_gowt, hs_dc_gowt, gs_dc_gowt, dice_mh_GOWT1])
x = ['None', 'G', 'M', 'H', 'GH', 'MH']
from nuclei_segmentation import visualisation

visualisation.comparison_preprocessing(dice_scores, x_label=x)
# Seaborn requires a dataframe
# dataframe = pd.DataFrame(data = np.transpose(dice_scores), columns=x)
# print(dataframe)
# ax = sns.swarmplot(data=dataframe,
#                    size=7,
#                    palette='magma_r')
# ax = sns.boxplot(showmeans=True,
#                  meanline=True,
#                  meanprops={'color': 'k', 'ls': '-', 'lw': 1},
#                  medianprops={'visible': False},
#                  whiskerprops={'visible': False},
#                  zorder=10,
#                  data=dataframe,
#                  showfliers=False,
#                  showbox=False,
#                  showcaps=False,
#                  ax=ax)
# ax.set(xlabel='Preprocessing',
#        ylabel='Dice Score',
#        title='Comparison of different preprocessing methods')
# plt.show()