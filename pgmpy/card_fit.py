from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

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
    'User_Has_CIP':         [None, 0, 1, 0, 0, 1, 0, 1],
    'User_Has_AMEX_P':      [0, 0, 0, 0, None, 1, 0, 1],
    'User_Has_AMEX_P_100K': [1, 0, None, 1, 0, 0, 0, 1],

    'Reach_5_24':           [1, 0, 0, 0, 0, 0, 1, 0],

    'Suggest_CIP':          [0, 1, 0, 1, 1, 0, 1, 0],
    'Suggest_AMEX_P':       [1, 0, 1, 1, 0, 0, 0, 1],
})


card_model = BayesianModel([
    ('User_Owns_5_Cards', 'Reach_5_24'),
    ('User_Reach_5_24', 'Reach_5_24'),
    ('Reach_5_24', 'Suggest_CIP'),
    ('User_Has_CIP', 'Suggest_CIP'),
    ('Reach_5_24', 'Suggest_AMEX_P'),
    ('User_Has_AMEX_P', 'Suggest_AMEX_P'),
    ('User_Has_AMEX_P_100K', 'Suggest_AMEX_P'),
])

card_model.fit(data)
print('=== Reach_5_24 ===')
print(card_model.get_cpds('Reach_5_24'))
print('\n')
'''
print('=== Suggest_CIP ===')
print(card_model.get_cpds('Suggest_CIP'))
print('\n')
print('=== Suggest_AMEX_P ===')
print(card_model.get_cpds('Suggest_AMEX_P'))
'''
print('\n')
print('= MaximumLikelihoodEstimator =')

cpd_reach_5_24 = MaximumLikelihoodEstimator(card_model, data).estimate_cpd('Reach_5_24')

print('=== Reach_5_24 ===')
print(cpd_reach_5_24)
print('\n')
suggest_cip = MaximumLikelihoodEstimator(card_model, data).estimate_cpd('Suggest_CIP')

print('=== Suggest_CIP ===')
print(suggest_cip)
print('\n')

suggest_amex_p = MaximumLikelihoodEstimator(card_model, data).estimate_cpd('Suggest_AMEX_P')

print('=== Suggest_AMEX_P ===')
print(suggest_amex_p)
print('\n')


'''
print('\n')
print('= BayesianEstimator =')

estimator = BayesianEstimator(card_model, data)
cpd_reach_5_24 = estimator.estimate_cpd(
    'Reach_5_24', prior_type="dirichlet", pseudo_counts=[2, 4])
print('=== Reach_5_24 ===')
print(cpd_reach_5_24)
print('\n')
'''

print('=== Inference ===')

card_infer = VariableElimination(card_model)

print('\n\ninfer: Suggest_CIP')
q = card_infer.query(
    variables=['Suggest_CIP'],
    evidence={
        'User_Owns_5_Cards': True, 
        'User_Reach_5_24': False, 
        'User_Has_CIP': False
    }
)
print(q.values)

print('\n\ninfer: Suggest_AMEX_P')
q = card_infer.query(
    variables=['Suggest_AMEX_P'],
    evidence={'User_Owns_5_Cards': True, 'User_Reach_5_24': False, 'User_Has_CIP': False}
)
print(q.values)