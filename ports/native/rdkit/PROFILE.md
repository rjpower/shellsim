# Native chemistry profile

This first RDKit 2025.9.3 producer builds the upstream molecule and SMILES parser
closure: SmilesParse, GraphMol, Geometry, DataStructs and RDGeneral. It uses
upstream Boost and RDKit's pinned Better Enums header. A libc++ compatibility
patch supplies the missing vector include in RealValueVect.h.

The producer configures this closure and exports its archives and source headers.
It omits the renderer, external font downloads, Python wrappers, serialization,
3D descriptors and optional chemistry plugins. The initial guest probe parses
ethanol and benzene, checks graph sizes and aromaticity, writes canonical SMILES
and rejects an unclosed ring. Other RDKit APIs and the Python package require
separate builds and qualification.
