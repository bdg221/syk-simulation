from workbench_algorithms.utils import PauliMask, PauliSum 
from math import floor 
from itertools import combinations
from .hamiltonian import syk_majorana_to_mask, SYK_hamil
from scipy.special import comb

from pytest import mark

@mark.parametrize(
    "coefficient, indices, pauli_string",
    [
        (-1, (0,1,2,3), 'Z0 Z1'),
        (-1, (4,5,6,7), 'Z2 Z3'),
        (-1, (0,1,12,13), 'Z0 Z6'),
        (-1, (1,2,3,4), 'X0 X2'),
        (-1, (7,8,9,10), 'X3 X5'),
        (-1, (0,1,7,8), 'Z0 X3 X4'),
        (1, (2,3,6,8), 'Z1 Y3 X4'),
        (1, (0,1,2,4), 'Z0 Y1 X2'),
        (-1, (2,4,6,8), 'Y1 X2 Y3 X4'),
        (-1, (0,6,12,14), 'Y0 Z1 Z2 X3 Y6 X7'),
        (1, (0,3,8,9), 'Y0 Y1 Z4'),
    ],
)

def test_jw_transform(coefficient, indices, pauli_string):
    pauli_op = syk_majorana_to_mask(indices)
    assert [pauli_op[0], pauli_op[1].get_pauli_string()] == [coefficient, pauli_string]

def test_syk_summands():
    for i in range(2,8):
        theory = comb(2*i, 4)
        hamil = SYK_hamil(2*i)
        assert theory == len(hamil)