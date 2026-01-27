from workbench_algorithms.utils import PauliMask, PauliSum 
from math import floor 
from itertools import combinations
import numpy as np

def SYK_pair_to_mask(p: int, q: int):
    """
    Function generating the PauliString for the product of two majorana fermions

    :param p: Index for the first majorana fermion
    :type p: int

    :param q: Index for the second majorana fermion
    :type q: int

    Output: List with coefficient in first entry and PauliMask in second
    """
    if q>p:    
        if floor(q/2) == floor(p/2):
            return [1j, PauliMask(0,2**floor(q/2))]
        else:
            x_mask = 2**floor(p/2)+2**floor(q/2)
            z_mask = sum(2**i for i in range(floor(p/2)+1, floor(q/2)))
            if q%2==1:
                z_mask += 2**floor(q/2)
            if p%2==0:
                z_mask += 2**floor(p/2)
                return [-1j, PauliMask(x_mask, z_mask)]
            else:
                return [1j, PauliMask(x_mask, z_mask)]
    else: 
        if floor(q/2) == floor(p/2):
            return [-1j, PauliMask(0,2**floor(q/2))]
        else:
            x_mask = 2**floor(q/2)+2**floor(p/2)
            z_mask = sum(2**i for i in range(floor(q/2)+1, floor(p/2)))
            if p%2==1:
                z_mask += 2**floor(p/2)
            if q%2==0:
                z_mask += 2**floor(q/2)
                return [1j, PauliMask(x_mask, z_mask)]
            else:
                return [-1j, PauliMask(x_mask, z_mask)]

def SYK_hamil(n: int, J: float=1, random_seed: int | None = None):
    """
    Function generating hamiltonian for the SYK model with 4-body interactions as a PauliSum
    We are using the convention that the coupling constant is 4!*J, where J is the coupling constant for all possible
    combinations, i.e., we only consider indices in increasing order

    :param n: Integer specifying number of qubits (i.e., 2n Majorana fermions)
    :type n: int

    :param J: coupling constant
    :type J: float

    :param random_seed(optional, default = none): Ability to set random_seed for testing/reproducability purposes
    :type n: int
    """

    if random_seed is not None:
        np.random.seed(random_seed)
    
    scale = np.sqrt(6/(2*n)**3)*J
    hamil = PauliSum()

    for tup in combinations(range(2*n),4):
        pauli_op1 = SYK_pair_to_mask(tup[0],tup[1])
        pauli_op2 = SYK_pair_to_mask(tup[2],tup[3])
        coef = np.random.normal(loc=0, scale= scale, size = 1)
        hamil.append([1/(96)*coef[0]*np.real(pauli_op1[0]*pauli_op2[0]), pauli_op1[1]*pauli_op2[1]])

    return hamil
