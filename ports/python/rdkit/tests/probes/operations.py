"""Exercise upstream molecule, file, fingerprint and NumPy binding behavior."""

import numpy as np
from rdkit import Chem, DataStructs, RDLogger, rdBase

assert rdBase.rdkitVersion == "2025.09.3"
vector = rdBase._vecti()
vector.append(2)
vector.append(5)
assert list(vector) == [2, 5]

ethanol = Chem.MolFromSmiles("CCO")
benzene = Chem.MolFromSmiles("c1ccccc1")
assert ethanol.GetNumAtoms() == 3 and ethanol.GetNumBonds() == 2
assert benzene.GetNumAtoms() == 6 and all(atom.GetIsAromatic() for atom in benzene.GetAtoms())
assert Chem.MolToSmiles(ethanol) == "CCO"
assert Chem.MolToSmiles(benzene) == "c1ccccc1"
assert ethanol.GetSubstructMatches(Chem.MolFromSmarts("[OX2H]")) == ((2,),)
assert Chem.AddHs(ethanol).GetNumAtoms() == 9
assert Chem.RemoveHs(Chem.AddHs(ethanol)).GetNumAtoms() == 3
restored = Chem.MolFromMolBlock(Chem.MolToMolBlock(ethanol))
assert Chem.MolToSmiles(restored) == "CCO"

fingerprint = Chem.RDKFingerprint(ethanol)
assert DataStructs.TanimotoSimilarity(fingerprint, fingerprint) == 1.0
adjacency = Chem.GetAdjacencyMatrix(ethanol)
assert isinstance(adjacency, np.ndarray)
np.testing.assert_array_equal(adjacency, [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
chiral = Chem.MolFromSmiles("F[C@](Cl)(Br)I")
Chem.AssignCIPLabels(chiral)
assert chiral.GetAtomWithIdx(1).GetProp("_CIPCode") in {"R", "S"}

RDLogger.DisableLog("rdApp.error")
assert Chem.MolFromSmiles("C1CC") is None
try:
    ethanol.GetAtomWithIdx(99)
except RuntimeError:
    pass
else:
    raise AssertionError("invalid atom index accepted")

print("RDKit molecules, files, substructures, fingerprints, NumPy and errors passed")
