import pandas as pd
import hedgehog as hh

bn = hh.BayesNet(
    ('Burglary', 'Alarm'),
    ('Earthquake', 'Alarm'),
    ('Alarm', 'John calls'),
    ('Alarm', 'Mary calls')
)

bn.P['Burglary'] = pd.Series({False: .999, True: .001})

bn.P['Earthquake'] = pd.Series({False: .998, True: .002})

bn.P['Alarm'] = pd.Series({
    (True, True, True): .95,
    (True, True, False): .05,
    (True, False, True): .94,
    (True, False, False): .06,
    (False, True, True): .29,
    (False, True, False): .71,
    (False, False, True): .001,
    (False, False, False): .999
})

bn.P['John calls'] = pd.Series({
    (True, True): .9,
    (True, False): .1,
    (False, True): .05,
    (False, False): .95
})

bn.P['Mary calls'] = pd.Series({
    (True, True): .7,
    (True, False): .3,
    (False, True): .01,
    (False, False): .99
})

bn.prepare()

print('Infer-1 ========')
result = bn.query('Burglary', event={'Mary calls': True, 'John calls': True})
print(type(result))
print(result)
print(result.keys())
print(result.to_dict())

print('Infer-2 ========')
result = bn.query('John calls', 'Mary calls', event={'Earthquake': True})
print(type(result))
print(result)
print(result.keys())
print(result.to_dict())