import numpy as np
import mpmath as mp
dtype = 'float64'
boltz = 8.617333262e-5 #eV/K

"""
Physical Quantities
"""
""" 
Funcion para calcular la probabilidad de cada uno de los estados
input: 
    - ee (numpy array): Arreglo con los valores de energia ordenados de menor a mayor
    - t (float): Valor de temperatura
    - pre (int): Entero positivo que indica la precision para calculos grandes
output:
    - Arreglo con las probabilidades asociadas a cada valor de energia
"""
def prob_states(ee: np.array, t: float, pre: int) -> np.array:
    np.seterr(all='raise')
    beta = 1.0/(t*boltz)
    ee_var = -ee*beta
    try:
        partition = np.exp( ee_var, dtype=dtype )
        Z = np.sum(partition, dtype=dtype)
        partition = np.divide( partition, Z, dtype=dtype )
    except FloatingPointError:
        with mp.workdps(pre):
            partition = [ mp.exp( e ) for e in ee_var ]
            Z = mp.fdiv( 1.0, mp.fsum(partition) )
            partition = [ float( mp.fmul(p, Z) ) for p in partition ]
    return np.round(np.array(partition),10)


"""
Funcion para calcular el logaritmo de la funcion de particion Z
input: 
    - ee (numpy array): Arreglo con los valores de energia ordenados de menor a mayor
    - t (float): Valor de temperatura
    - pre (int): Entero positivo que indica la precision para calculos grandes
output:
    - Logaritmo natural de Z
"""
def log_z_function(ee: np.array, t: float, pre: int) -> np.array:
    np.seterr(all='raise')
    beta = 1.0/(t*boltz)
    ee_var = -ee*beta
    try:
        partition = np.exp( ee_var, dtype=dtype )
        Z = np.sum(partition, dtype=dtype)
        Z = np.log(Z)
    except FloatingPointError:
        with mp.workdps(pre):
            partition = [ mp.exp( e ) for e in ee_var ]
            Z = mp.fsum(partition)
            Z = mp.log(Z)
    return float(Z)

"""
Funcion para calcular el calor especifico
input:
    - ee (numpy array): Arreglo con los valores de energia ordenados de menor a mayor
    - proy (numpy array): Arreglo con las proyecciones de los estados sobre el operador (producto braket)
    - t (float): Valor de temperatura
    - pre (int): Entero positivo que indica la precision para calculos grandes
output:
    - Valor del calor especifico en una temperatura especifica
"""
def specific_heat(ee: np.array, proy: np.array, t: float, pre: int) -> float:
    partition = prob_states(ee, t, pre)
    entalpia = np.sum( partition*ee, dtype=dtype )
    entalpia_2 = np.sum( partition*(ee**2), dtype=dtype )
    tmp_var = entalpia_2 - (entalpia**2)
    return np.divide( tmp_var, (t*t*boltz) )


"""
Funcion para calcular la entropia
input:
    - ee (numpy array): Arreglo con los valores de energia ordenados de menor a mayor
    - proy (numpy array): Arreglo con las proyecciones de los estados sobre el operador (producto braket) 
    - t (float): Valor de temperatura
    - pre (int): Entero positivo que indica la precision para los calculos
output:
   - Valor de la entropia en una temperatura especifica 
"""
def entropy(ee: np.array, proy: np.array, t: float, pre: int) -> float:
    partition = prob_states(ee, t, pre)
    thermal = np.sum( partition*ee, dtype=dtype )
    free_energy = -boltz*log_z_function(ee, t, pre)
    return np.divide(thermal, t, dtype=dtype) - free_energy


"""
Funcion generica para calcular el valor esperado
input:
    - ee (numpy array): Arreglo con los valores de energia ordenados de menor a mayor
    - proy (numpy array): Arreglo con las proyecciones de los estados sobre el operador (producto braket)
    - t (float): Valor de temperatura
    - pre (int): Entero positivo que indica la precision para los calculos
output:
    - Valor del valor esperado de un operador a una temperatura dada
"""
def expected_value(ee: np.array, proy: np.array, t: float, pre: int) -> float:
    partition = prob_states(ee, t, pre)
    return np.sum( partition*proy, dtype=dtype )


"""
Funcion para calcular la variacion de entropia entre dos puntos
input:
    - ee_initial (numpy array): Arreglo con los valores de energia del punto inicial
"""
def isothermal_entropy_change(ee_initial: np.array, ee_final:np.array, t: float, pre: int) -> float:
    partition_initial = prob_states(ee_initial, t, pre)
    partition_final = prob_states(ee_final, t, pre)

    thermal_initial = partition_initial*ee_initial
    thermal_final = partition_final*ee_final

    tmp_var1 = (thermal_initial -  thermal_final)/t
    tmp_var2 = -boltz*( log_z_function(ee_final, t, pre) - log_z_function(ee_initial, t, pre) )
    return tmp_var1 + tmp_var2



