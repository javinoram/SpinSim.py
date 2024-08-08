import spinsim as ss
import numpy as np

def test_prob_distribution():
    energies = np.array([-1e-5, 1e-5, 1e-5, 1e-5])
    t = 0.1
    dist = ss.thermodynamic.distributions.distribution(energies, t, 120)
    dist_theo = np.array( [0.77246653, 0.07584449, 0.07584449, 0.07584449] )
    
    diff = np.round( np.abs( dist - dist_theo ), 7 )
    diff = np.sum(diff)/4.0
    if diff < 1e-7:
        assert True
    else:
        assert False


def test_log_z_function():
    energies = np.array([-1e-5, 1e-5, 1e-5, 1e-5])
    t = 0.1
    function_value = ss.thermodynamic.distributions.log_z_function(energies, t, 120)
    if np.abs( function_value-1.418618409 ) < 1e-7:
        assert True
    else:    
        assert False


def test_der_prob_distribution():
    energies = np.array([-1e-5, 1e-5, 1e-5, 1e-5])
    t = 0.1
    dist = ss.thermodynamic.distributions.derivate_distribution(energies, t, 120)
    
    tmp_val = -6.323685603
    val1 = (11.60451812 - tmp_val)*0.075844489
    val2 = (-11.60451812 - tmp_val)*0.77246653
    dist_theo = np.array([val2, val1, val1, val1])

    diff = np.round( np.abs( dist - dist_theo ), 7 )
    diff = np.sum(diff)/4.0
    if diff < 1e-7:
        assert True
    else:
        assert False


