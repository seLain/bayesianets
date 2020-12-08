from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.estimators import BDeuScore, K2Score, BicScore
from pgmpy.estimators import ExhaustiveSearch

'''
= Decision Nodes =
User_Owns_5_Cards
User_Reach_5_24
User_Has_CIP
User_Has_AMEX_P
User_Has_AMEX_P_100K
= Chance Nodes =
Reach_5_24
= Utility Nodes =
Suggest_CIP
Suggest_AMEX_P
'''

data = pd.DataFrame(data={
    'User_Owns_5_Cards':    [1, 1, 0, 1, None, 0, 1, None],
    'User_Reach_5_24':      [1, 0, 0, 1, 0, 0, 1, None],

    'Reach_5_24':           [1, 0, 0, 0, 0, 0, 1, 0],

})

bic = BicScore(data)
es = ExhaustiveSearch(data, scoring_method=bic)
best_model = es.estimate()
print(best_model.edges())

print("\nAll DAGs by score:")
for score, dag in reversed(es.all_scores()):
    print(score, dag.edges())