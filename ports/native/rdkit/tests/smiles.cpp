// Exercise the actual upstream molecule graph and parser before Python wrappers.
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <memory>

int main() {
    std::unique_ptr<RDKit::ROMol> ethanol(RDKit::SmilesToMol("CCO"));
    if (!ethanol || ethanol->getNumAtoms() != 3 || ethanol->getNumBonds() != 2)
        return 1;
    if (RDKit::MolToSmiles(*ethanol) != "CCO") return 2;
    std::unique_ptr<RDKit::ROMol> benzene(RDKit::SmilesToMol("c1ccccc1"));
    if (!benzene || benzene->getNumAtoms() != 6) return 3;
    for (const auto atom : benzene->atoms())
        if (!atom->getIsAromatic()) return 4;
    std::unique_ptr<RDKit::ROMol> invalid(RDKit::SmilesToMol("C1CC"));
    return invalid ? 5 : 0;
}
