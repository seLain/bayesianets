from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


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


card_model = BayesianModel([
    ('User_Owns_5_Cards', 'Reach_5_24'),
    ('User_Reach_5_24', 'Reach_5_24'),
    ('Reach_5_24', 'Suggest_CIP'),
    ('User_Has_CIP', 'Suggest_CIP'),
    ('Reach_5_24', 'Suggest_AMEX_P'),
    ('User_Has_AMEX_P', 'Suggest_AMEX_P'),
    ('User_Has_AMEX_P_100K', 'Suggest_AMEX_P'),
])


cpd_user_owns_5_cards = TabularCPD(
    variable='User_Owns_5_Cards', variable_card=2,
    values=[[0.5], [0.5]])

print('cpd_user_owns_5_cards')
print(cpd_user_owns_5_cards)

cpd_user_reach_5_24 = TabularCPD(
    variable='User_Reach_5_24', variable_card=2,
    values=[[0.5], [0.5]])

print('cpd_user_reach_5_24')
print(cpd_user_reach_5_24)

cpd_user_has_cip = TabularCPD(
    variable='User_Has_CIP', variable_card=2,
    values=[[0.5], [0.5]])

cpd_user_has_amex_p = TabularCPD(
    variable='User_Has_AMEX_P', variable_card=2,
    values=[[0.5], [0.5]])

cpd_user_has_amex_p_100k = TabularCPD(
    variable='User_Has_AMEX_P_100K', variable_card=2,
    values=[[0.5], [0.5]])

cpd_reach_5_24 = TabularCPD(
    variable='Reach_5_24', variable_card=2,
    values=[
        [1, 0.9, 0.7, 0], # False
        [0, 0.1, 0.3, 1], # True
    ],
    evidence=['User_Owns_5_Cards', 'User_Reach_5_24'],
    evidence_card=[2, 2])

print('cpd_reach_5_24')
print(cpd_reach_5_24)

cpd_suggest_cip = TabularCPD(
    variable='Suggest_CIP', variable_card=2,
    values=[
        [0, 0.9, 0.8, 1], # False
        [1, 0.1, 0.2, 0], # True
    ],
    evidence=['Reach_5_24', 'User_Has_CIP'],
    evidence_card=[2, 2]
)

print('cpd_suggest_cip')
print(cpd_suggest_cip)

cpd_suggest_amex_p = TabularCPD(
    variable='Suggest_AMEX_P', variable_card=2,
    values=[
        [0.9, 0.3, 1, 1, 0.9, 0, 1, 1], # False
        [0.1, 0.7, 0, 0, 0.1, 1, 0, 0], # True
    ],
    evidence=['Reach_5_24', 'User_Has_AMEX_P', 'User_Has_AMEX_P_100K'],
    evidence_card=[2, 2, 2]
)

card_model.add_cpds(
    cpd_user_owns_5_cards, cpd_user_reach_5_24, cpd_user_has_cip, 
    cpd_user_has_amex_p, cpd_user_has_amex_p_100k, cpd_reach_5_24,
    cpd_suggest_cip, cpd_suggest_amex_p
)

card_model.check_model()

print('=== Statistical inference ===')

card_infer = VariableElimination(card_model)

print('\n\ninfer: Reach_5_24')
q = card_infer.query(
    variables=['Reach_5_24'],
    evidence={
        'User_Owns_5_Cards': True, 
        'User_Reach_5_24': False, 
        'User_Has_CIP': False
    }
)
#print(type(q))
#print(q.variables)
#print(q.__dict__)
#print(type(q.values))
print(q.values)


print('\n\ninfer: Suggest_CIP')
q = card_infer.query(
    variables=['Suggest_CIP'],
    evidence={
        'User_Owns_5_Cards': True, 
        'User_Reach_5_24': False, 
        'User_Has_CIP': False
    }
)
#print(type(q))
#print(q.variables)
#print(q.__dict__)
#print(type(q.values))
print(q.values)


print('\n\ninfer: Suggest_AMEX_P')
q = card_infer.query(
    variables=['Suggest_AMEX_P'],
    evidence={'User_Owns_5_Cards': True, 'User_Reach_5_24': False, 'User_Has_CIP': False}
)
#print(type(q))
#print(q.variables)
#print(q.__dict__)
#print(type(q.values))
print(q.values)