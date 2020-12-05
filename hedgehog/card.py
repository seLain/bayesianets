import pandas as pd
import hedgehog as hh

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

bn = hh.BayesNet(
    (['User_Owns_5_Cards', 'User_Reach_5_24'], 'Reach_5_24'),
    (['Reach_5_24', 'User_Has_CIP'], 'Suggest_CIP'),
    (['Reach_5_24', 'User_Has_AMEX_P', 'User_Has_AMEX_P_100K'], 'Suggest_AMEX_P')
)

bn.P['User_Owns_5_Cards'] = pd.Series({False: .5, True: .5})
bn.P['User_Reach_5_24'] = pd.Series({False: .5, True: .5})
bn.P['User_Has_CIP'] = pd.Series({False: .5, True: .5})
bn.P['User_Has_AMEX_P'] = pd.Series({False: .5, True: .5})
bn.P['User_Has_AMEX_P_100K'] = pd.Series({False: .5, True: .5})

bn.P['Reach_5_24'] = pd.Series({
    (True, True, True): .1,
    (True, True, False): .9,
    (True, False, True): .3,
    (True, False, False): .7,
    (False, True, True): .1,
    (False, True, False): .9,
    (False, False, True): 0,
    (False, False, False): 1
})

bn.P['Suggest_CIP'] = pd.Series({
    (True, True, True): 0,
    (True, True, False): 1,
    (True, False, True): .2,
    (True, False, False): .8,
    (False, True, True): .1,
    (False, True, False): .9,
    (False, False, True): 1,
    (False, False, False): 0
})

bn.P['Suggest_AMEX_P'] = pd.Series({
    (True, True, True, True): 0,
    (True, True, True, False): 1,
    (True, True, False, True): 0,
    (True, True, False, False): 1,
    (True, False, True, True): 1,
    (True, False, True, False): 0,
    (True, False, False, True): .1,
    (True, False, False, False): .9,
    (False, True, True, True): 0,
    (False, True, True, False): 1,
    (False, True, False, True): 0,
    (False, True, False, False): 1,
    (False, False, True, True): .7,
    (False, False, True, False): .3,
    (False, False, False, True): .1,
    (False, False, False, False): .9
})

bn.prepare()

print('Infer-1 ========')
result = bn.query('Suggest_CIP', event={'User_Owns_5_Cards': True, 'User_Reach_5_24': False, 'User_Has_CIP': False})
print(type(result))
print(result)
print(result.keys())
print(result.to_dict())