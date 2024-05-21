import numpy as np
import mpmath as mp
np.seterr(all='raise') 
dtype = 'float64'
boltz = 8.617333262e-5 #eV/K



"""
Funcion para calcular la entropia de von Neumann para un arreglo de valores propios positivos no nulos
input:
    e: Vector de valores propios positivos no nulos.
    pre: Entero positivo que indica la precision para calculos grandes
output:
    vn: entropia de von neumann
""" 
def von_neumann_entropy(ee: np.array, pre: int) -> float:
    ee = ee[ ee>0 ]
    try:
        vn = np.sum( -ee*np.log(ee, dtype=dtype), dtype=dtype )
    except FloatingPointError:
        with mp.workdps(pre):
            vn = mp.fsum( [ -e*mp.log( e ) for e in ee ] )
    return vn

"""
Funcion para calcular la concurrencia de un sistema de multiples particulas, usando la siguiente
formula C_M(rho)=sqrt( 2(1 - Tr[ rho_M**2 ] ) ) donde M es una particion dada por la aplicacion de
la traza parcial.
input:
    rho: Matriz de densidad reducida
output:
    Valor de la concurrencia en la particion dada
"""
def concurrence(rho: np.array) -> float:
    density_val = np.trace( rho@rho )
    return np.sqrt( 2*(1 - density_val) )



"""
Funcion para calcular la traza parcial de una matriz cuadrada, de izquierda a derecha
input:
    rho: Matriz densidad
    spin_list: Lista con los valores del spin en cada sitio
    number_spines: Numero de espines que se quiere eliminar del sistema
output:
    sub_system: matriz resultante de aplicar la traza parcial
""" 
def partial_trace_lr(rho: np.array, spin_list: list, number_spines: int) -> np.array:    
    for i in range( number_spines ):
        s = int(2*spin_list[i] + 1)
        rho = np.sum( [rho[ s*j: s*(j+1), s*j:s*(j+1)] for j in range( int(rho.shape[0]/s) )], axis=0 )
    return rho



"""
Funcion que permite calcular la fidelidad entre dos estados, F(s1, s2)=|s1@s2|**2
El resultado de la funcion esta acotado entre 0 y 1, siendo 0 un indicador de que son
perpendiculares y 1 que son iguales.
input:
    state1: Vector columna del primer estado.
    state2: Vector columna del segundo estado.
output:
    Valor de la similitud entre los estados
"""
def fidelity_states( state1: np.array, state2: np.array ) -> float:
    val1 = state1.conj().T*state2
    val2 = state2.conj().T*state1
    return val1*val2


"""
Funcion para revisar si el estado de minima energia esta degenerado y ver cuantos estados
hay que considerar
input:
    ee: Arreglo con las energias del sistema, ordenados de menor a mayor
    tol: Tolerancia respecto a errores numericos de las energia
output:
    Cantidad de estados a consirar, en caso de error se retorna -1
"""
def count_rep_gs(ee: np.array, tol: float) -> int:
    for i, ee in enumerate(ee):
        if np.abs(ee - ee[0])>tol:
            return i+1
    return -1



"""
Workflows
"""
"""
Funcion que permite calcular la entropia de von Neumann de la matriz de densidad del estado de minima energia siguiendo 
una topologia lineal. Para entender el proceso, considera que los espines son ordenados de forma lineal siguiendo la
enumeracion ingresada y en base a ella se procede a aplicar una traza parcial hasta que no quede mas sistema.
input:
    Op: Operador al que se le va a calcular la entropia en cada traza parcial.
    spin_list: Lista con el valor de los espines en cada sito
output:
    Lista con los valores de la entropia en cada aplicacion de la traza que van desde no aplicarla hasta que no quede nada.
"""
def entanglement_entropy_per_site_gs(Op: np.array, spin_list: list) -> list:
    ee, vv = np.linalg.eigh(Op)
    cant = count_rep_gs(ee, 1e-7)

    density = np.zeros( Op.shape )
    for i in range(cant):
        state = np.array( [[j] for j in vv[:,i]] )
        state_dag = state.conj().T
        density += state.dot( state_dag )
    density = ( 1/cant )*np.real( density )
    return [ von_neumann_entropy( np.linalg.eigvalsh( partial_trace_lr( density, spin_list, i )  ), 90) for i in range( len(spin_list) ) ]
