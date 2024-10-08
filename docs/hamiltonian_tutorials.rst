======================
Hamiltonian tutorial
======================

The idea of this tutorial is ilustrate how to create a matrix represation of a hamiltonian using this package.
As it's expected, the hamiltonian that we consider are the one of spin models (the operators are the product of Pauli matrices of a given values of spin).


The Model
---------
Before we start using the package let introduce a example model (we will use it in all the tutorial). Let consider a triangle, this one is modeled as follow.

.. math::
  \mathcal{H} = J_1S_0S_1 + J_2S_1S_2 + J_3S_0S_3

The operator is defined as :math:`S_{i} = (\sigma_{x,i}, \sigma_{y,i}, \sigma_{z,i})`, where the :math:`\sigma_{i,j}` is the pauli matrices of a given spin in the site :math:`i`.
The product of this operators is a dot product, and the product in the resulting sum is a multiplication between the pauli matrices on different site. For example:

.. math::
   \sigma_{x,0}\sigma_{x,1} = (\sigma_{x,0} \otimes \mathbf{I} \otimes \mathbf{I})(\mathbf{I} \otimes \sigma_{x,1} \otimes \mathbf{I})

Its a simple matrix multiplication, but the pauli matrices are in different places.


Computing the paulices matrices of any spin
--------------------------------------------
The next point is how we can create the matrices of any given spin, to do that, we use the following formulas. For each matrix we have the following 
restriction :math:`1 \leq i \leq 2s + 1` and :math:`1 \leq j \leq 2s + 1`, and s being the spin value.


For the :math:`\sigma_x` we have that each element of the matrix follows the next equation:

.. math::
   (S_x)_{i,j} = 0.5( \delta_{i,j+1} + \delta_{i+1,j} )\sqrt{ (s+1)(i+j-1) - ij }

For the :math:`\sigma_y` we have that each element of the matrix follows the next equation:

.. math::
   (S_y)_{i,j} = 0.5i( \delta_{i,j+1} - \delta_{i+1,j} )\sqrt{ (s+1)(i+j-1) - ij }


For the :math:`\sigma_z` we have that each element of the matrix follows the next equation:

.. math::
  (S_z)_{i,j} = (s + 1 - i)\delta_{i,j}


Important elements
------------------
Now that we know how some to compute some things, let list the important elements. The first one, we need to have a enumeration of the sites of the system to 
indicate the value of the spin of that site, so, we need to create a list with the values of the spin. The second one is a list of tuples that indicate 
which site is conected with other one (it's important that the first index is lower or equal that the second one). And finally, a list with the values of the exchange. 

Taking the :math:`\mathcal{H}` model, the three element should look as [0.5, 0.5, 0.5], [(0,1), (1,2), (0,2)] and [:math:`J_1`, :math:`J_2`, :math:`J_3`].


Coding part
-----------
Now that we have the enough information, here you can find an example of how the code should looks like.

.. code-block:: python

   from spinsim.operators import set_sij_vector
   from spinsim.hamiltonian import construct_hamiltonian

   indexs = [ (0,1), (1,2), (0,2) ]
   exchanges = [J1, J2, J3]
   spins = [0.5, 1.0, 0.5]

   SiSjvectors = set_sij_vector(indexs, 3) 

   terms = []
   for J, op in zip(exchanges, SiSjvectors):
    terms += [ [J, op[0]], [J, op[1]], [J, op[2]] ] 

   H = construct_hamiltonian(terms, spins)


After defining the basis elements (indexs, exchanges and spins) we use the *set_sij_vector()* to construct the set of vectors with the product :math:`S_iS_j`, 
the list had a size equal to the list of indexs. After that we create the list where each term has a exchange asociated. Finally using *construct_hamiltonian()* 
we get the matrix representation of the system.

