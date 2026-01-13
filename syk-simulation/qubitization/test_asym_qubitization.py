from pytest import mark
import numpy as np

import importlib

aq_module = importlib.import_module("syk-simulation.qubitization.asymmetric_qubitization")


# from "syk-simulation".qubitization.asymmetric_qubitization import OracleA, OracleB, AsymmetricQubitization
from psiqworkbench import QPU, Qubits


# @mark.parametrize(
#     "N,num_terms,depth",
#     [
#         # (6, 12, 2),
#         # (8, 70, 5),
#     ],
# )  # (10, 100, 7)
def manual_test_asymmetric_qubitization(N, num_terms, depth):
    # N = 8  # number of Majorana fermions, 100 is interesting
    # num_terms = 70  # number of Pauli terms in the Hamiltonian
    # depth = 5

    num_system_qubits = int(np.ceil(N / 2))
    num_index_qubits = int(np.ceil(np.log2(num_terms)))

    total_qubits = int(num_system_qubits + num_index_qubits + 1)
    qpu = QPU(num_qubits=total_qubits, filters=[">>unitary>>", ">>buffer>>"])

    branch = Qubits(1, "branch", qpu=qpu)
    index = Qubits(num_index_qubits, "index", qpu=qpu)
    system = Qubits(num_system_qubits, "system", qpu=qpu)

    AQ = aq_module.AsymmetricQubitization()

    pauliStrings = generate_pauli_strings(num_terms, num_system_qubits)
    AQ.compute(branch=branch, index=index, system=system, depth=depth, terms=pauliStrings)

    # Here you would add assertions to verify the correctness of the qubitization
    # For example, checking the final state vector or specific gate applications
    # This is a placeholder for actual verification logic
    assert True  # Replace with actual checks


def generate_pauli_strings(num_terms, num_system_qubits):
    pauli_strings = []
    paulis = ["I", "X", "Y", "Z"]
    for _ in range(num_terms):
        term = "".join(np.random.choice(paulis, size=num_system_qubits))
        pauli_strings.append(" ".join(f"{p}{i}" for i, p in enumerate(term)))
    return pauli_strings


# def run_aq():
#     manual_test_asymmetric_qubitization(6, 12, 2)


# run_aq()
