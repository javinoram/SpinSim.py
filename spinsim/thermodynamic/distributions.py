import numpy as np
import mpmath as mp
boltz = 8.617333262e-5 #eV/K

def distribution(ee: np.array, t: float, pre: int, dtype='float64') -> np.array:
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


def log_z_function(ee: np.array, t: float, pre: int, dtype='float64') -> np.array:
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


def derivate_distribution(ee: np.array, t: float, pre: int, dtype='float64') -> np.array:
    np.seterr(all='raise')
    beta1 = 1.0/(t*boltz)
    beta2 = 1.0/(t*t*boltz)

    partition = np.exp( -ee*beta1, dtype=dtype )
    Z = np.sum(partition, dtype=dtype)
    probs = np.divide( partition, Z, dtype=dtype )

    ee_sum_tmp = []
    for e,p in zip(ee, probs):
        ee_sum_tmp.append( p*e*beta2 )
    ee_sum_tmp = np.sum(np.array( ee_sum_tmp ))

    final = []
    for e,p in zip(ee, probs):
        val = p*( e*beta2 - ee_sum_tmp)
        final.append(val)

    return np.array(final)
