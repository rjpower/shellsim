# RDKit chemistry profile

This port builds the upstream RDKit 2025.9.3 source and Python package. Its
mandatory Chemistry imports use rdBase, cDataStructs, rdGeometry, rdchem,
rdCIPLabeler, rdmolfiles and rdmolops. Their C++ chemistry implementation lives
in one shared native provider; Boost.Python and Boost.NumPy also use shared
providers so extensions share their type registry.

The guest probe covers molecule and aromatic graph construction, canonical
SMILES, SMARTS substructure matching, hydrogen addition/removal, MolBlock
round-trips, fingerprints, NumPy adjacency matrices, stereochemistry and
invalid input using the actual upstream wrappers in the guest.

Optional InChI, rendering, embedding, force-field, descriptor and reaction
Python modules and data-dependent APIs are outside this initial profile;
the wheel does not ship the upstream data files. The native closure includes
the dependencies required by the selected wrappers. Guest execution remains
subject to the simulator's resource limits and process control.
The existing SDK signal bridge reports ENOTSUP for custom C++ SIGINT handlers;
long calculations remain bounded by simulator CPU limits.
