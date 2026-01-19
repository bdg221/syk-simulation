from workbench_algorithms.utils import PauliMask, PauliSum 
from math import floor 
from itertools import combinations
import numpy as np


def syk_majorana_to_mask(array: tuple | list):
    '''
    Function returning PauliMask corresponding to given product of Majorana fermions
    
    :param array: list in ascending order specifying the Majorana fermions
    :type array: tuple | list
    '''

    array = list(array)
    if array != sorted(array):
        raise Exception("Indices must be in ascending order.")
    if len(set(array))!=4:
        raise Exception("Indices must be unique.")
    
    coefs = [floor(i/2) for i in array]

    z_mask=0
    x_mask=0
    value=1

    if coefs[1] == coefs[2]:
        value*=-1

        if range(coefs[0]+1, coefs[3]):
            z_mask += sum([2**i for i in range(coefs[0]+1, coefs[1])])
            z_mask += sum([2**i for i in range(coefs[1]+1, coefs[3])])
        else:
            z_mask += 2**coefs[1]
        
        x_mask += 2**coefs[0] + 2**coefs[3]
        if array[0]%2==0:
            z_mask += 2**coefs[0]
            value*=-1
        if array[3]%2==1:
            z_mask += 2**coefs[3]

    else:
        if range(coefs[0], coefs[1]):
            value*=1j
            z_mask += sum([2**i for i in range(coefs[0]+1, coefs[1])])
            x_mask += 2**coefs[0]+2**coefs[1]
            if array[0]%2==0:
                z_mask +=2**coefs[0]
                value*=-1
            if array[1]%2==1:
                z_mask +=2**coefs[1]
        elif coefs[0]==coefs[1]:
            z_mask += 2**coefs[0]
            value *= 1j
        else:
            x_mask += 2**coefs[0]+2**coefs[1]
            if array[0]%2==1:
                z_mask += 2**coefs[0]
            if array[1]%2==1:
                z_mask += 2**coefs[1]
    
        if range(coefs[2], coefs[3]):
            value*=1j
            z_mask += sum([2**i for i in range(coefs[2]+1, coefs[3])])
            x_mask += 2**coefs[2]+2**coefs[3]
            if array[2]%2==0:
                z_mask +=2**coefs[2]
                value*=-1
            if array[3]%2==1:
                z_mask +=2**coefs[3]    
        elif coefs[2]==coefs[3]:
            z_mask += 2**coefs[2]
            value *= 1j
        else:
            x_mask += 2**coefs[2]+2**coefs[3]
            if array[2]%2==1:
                z_mask += 2**coefs[2]
            if array[3]%2==1:
                z_mask += 2**coefs[3]
   
    return [value, PauliMask(x_mask, z_mask)]


def SYK_hamil(n: int, J: float=1, random_seed: int | None = None):
    """
    Function generating hamiltonian for the SYK model with 4-body interactions as a PauliSum
    We are using the convention that the coupling constant is 4!*J where J is the coupling constant for all possible
    combinations

    :param n: Even integer specifying number of Majorana fermions
    :type n: int

    :param J: coupling constant
    :type J: float
    """

    if random_seed is not None:
        np.random.seed(random_seed)
    
    scale = np.sqrt(6/n**3)*J
    hamil = PauliSum()

    for tup in combinations(range(n),4):
        pauli_op = syk_majorana_to_mask(tup)
        coef = np.random.normal(loc=0, scale= scale, size = 1)
        hamil.append([1/4*coef[0]*pauli_op[0], pauli_op[1]])

    return hamil
