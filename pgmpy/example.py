from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

import time

cancer_model = BayesianModel([
    ('Pollution', 'Cancer'),
    ('Smoker', 'Cancer'),
    ('Cancer', 'Xray'),
    ('Cancer', 'Dyspnoea')
])


cpd_poll = TabularCPD(
    variable='Pollution', variable_card=2,
    values=[[0.9], [0.1]])

cpd_smoke = TabularCPD(
    variable='Smoker', variable_card=2,
    values=[[0.3], [0.7]])

cpd_cancer = TabularCPD(
    variable='Cancer', variable_card=2,
    values=[
        [0.03, 0.05, 0.001, 0.02], 
        [0.97, 0.95, 0.999, 0.98]
    ],
    evidence=['Smoker', 'Pollution'],
    evidence_card=[2, 2])

cpd_xray = TabularCPD(
    variable='Xray', variable_card=2,
    values=[
        [0.9, 0.2],
        [0.1, 0.8]
    ],
    evidence=['Cancer'],
    evidence_card=[2]
)

cpd_dysp = TabularCPD(
    variable='Dyspnoea', variable_card=2,
    values=[
        [0.65, 0.3],
        [0.35, 0.7]
    ],
    evidence=['Cancer'],
    evidence_card=[2]
)

cancer_model.add_cpds(cpd_poll, cpd_smoke, cpd_cancer, cpd_xray, cpd_dysp)

cancer_model.check_model()

print('=== Statistical inference ===')

cancer_infer = VariableElimination(cancer_model)

q = cancer_infer.query(variables=['Xray', 'Dyspnoea'], evidence={'Pollution': True, 'Smoker': False})

q = cancer_infer.query(variables=['Xray'], evidence={'Pollution': True, 'Smoker': False})

q = cancer_infer.query(variables=['Cancer'], evidence={'Pollution': True, 'Smoker': False})
print(type(q))
print(q.variables)
print(q.__dict__)
print(type(q.values))
print(q.values)

q = cancer_infer.query(variables=['Cancer'], evidence={'Pollution': True, 'Smoker': True})
print(type(q))
print(q.variables)
print(q.__dict__)
print(type(q.values))
print(q.values)

q = cancer_infer.query(variables=['Cancer'], evidence={'Pollution': False, 'Smoker': False})
print(type(q))
print(q.variables)
print(q.__dict__)
print(type(q.values))
print(q.values)