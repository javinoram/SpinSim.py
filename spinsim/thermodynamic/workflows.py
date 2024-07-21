from .compute_observables import specific_heat, entropy, valor_esperado
import numpy as np


"""
WORKFLOWS
"""
""" 
Funcion que calcula el calor especifico para un conjunto de temperaturas
input: 
    - op (numpy array): Operador hermitiano al que se le quiere calcular el
    calor especifico
    - temp (numpy array): Arreglo con las temperaturas
    - pre (int): Entero positivo que indica la precision para calculos grandes
output:
    - Valor del calor especifico en cada temperatura
"""
def specific_heat_workflow(op: np.array, temp: np.array, pre: int) -> np.array:
    ee = np.linalg.eigvalsh(op)
    return np.array( [ specific_heat(ee, None, t, pre) for t in temp ] )


""" 
Funcion que calcula la entropia para un conjunto de temperaturas
input: 
    - op (numpy array): Operador hermitiano al que se le quiere calcular la entropia
    - temp (numpy array): Arreglo con las temperaturas
    - pre (int): Entero positivo que indica la precision para calculos grandes
output:
    - Valor de la entropia en cada temperatura
"""
def entropy_workflow(op: np.array, temp: np.array, pre: int) -> np.array:
    ee = np.linalg.eigvalsh(op)
    return np.array( [ entropy(ee, None, t, pre) for t in temp ] )


""" 
Funcion que calcula el valor esperado de un operador
input: 
    - op_base (numpy array): Operador hermitiano al que se le toman los valores y vectores propios
    - operator (numpy array): Operador al que se le calcula el valor esperado
    - temp (numpy array): Arreglo con las temperaturas
    - pre (int): Entero positivo que indica la precision para calculos grandes
output:
    - Arreglo de los valores esperados a diferentes temperaturas
"""
def expected_value_workflow(op_base: np.array, operator: np.array, temp: np.array, pre: int) -> np.array:
    ee, vv = np.linalg.eigh(op_base)
    proy = np.array( [ ((vv[:,k]).T.conj()).dot(operator).dot(vv[:,k]) for k in range(len(ee))] )
    return np.array( [ valor_esperado(ee, proy, t, pre) for t in temp ] )



