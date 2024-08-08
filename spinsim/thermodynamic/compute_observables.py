import numpy as np
from spinsim.thermodynamic.distributions import distribution, derivate_distribution, log_z_function

boltz = 8.617333262e-5 #eV/K
nub = 5.7883818066e-5 #eV/T

def specific_heat(ee: np.array, t: float, pre: int) -> float:
    partition = distribution(ee, t, pre, 'float64')
    entalpia = partition@ee
    entalpia_2 = partition@(ee**2)
    tmp_var = entalpia_2 - (entalpia**2)
    return np.divide( tmp_var, (t*t*boltz) )


def entropy(ee: np.array, t: float, pre: int) -> float:
    partition = distribution(ee, t, pre, 'float64')
    thermal = partition@ee
    free_energy = -boltz*log_z_function(ee, t, pre, 'float64')
    return np.divide(thermal, t ) - free_energy


def expected_value(ee: np.array, proy: np.array, t: float, pre: int) -> float:
    partition = distribution(ee, t, pre, 'float64')
    return partition@proy


def isothermal_entropy_change(ee_initial: np.array, ee_final:np.array, t: float, pre: int) -> float:
    partition_initial = distribution(ee_initial, t, pre, 'float64')
    partition_final = distribution(ee_final, t, pre, 'float64')

    thermal_initial = partition_initial@ee_initial
    thermal_final = partition_final@ee_final

    tmp_var1 = (thermal_initial -  thermal_final)/t
    tmp_var2 = -boltz*( log_z_function(ee_final, t, pre, 'float64') - log_z_function(ee_initial, t, pre, 'float64') )
    return tmp_var1 + tmp_var2


def magnetization(ee: np.array, proy: np.array, t: float, gyro: float, pre: int) -> float:
    partition = distribution(ee, t, pre, 'float64')
    valor = gyro*nub*( partition@proy )
    return valor


def gruneisen_ratio(ee: np.array, proy: np.array, t: float, gyro: float, pre: int) -> float:
    valor = 0.0
    ders = derivate_distribution(ee, t, pre, 'float64')
    derivada_mag = np.round( np.sum( ders*proy ), 10)

    calor = np.round( specific_heat(ee, t, pre), 10)
    if np.abs( calor ) < 1e-8:
        return np.inf
    if np.abs( derivada_mag ) < 1e-8:
        return 0.0

    valor = np.divide( derivada_mag, valor )
    return -nub*gyro*valor

