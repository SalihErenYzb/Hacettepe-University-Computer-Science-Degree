import java.util.*;

// Class representing molecular data
public class MolecularData {

    // Private fields
    private final List<Molecule> molecules; // List of molecules

    // Constructor
    public MolecularData(List<Molecule> molecules) {
        this.molecules = molecules;
    }

    // Getter for molecules
    public List<Molecule> getMolecules() {
        return molecules;
    }

    // Method to identify molecular structures
    // Return the list of different molecular structures identified from the input data

    public void dfs(Molecule molecule, HashSet<String> visited, MolecularStructure structure) {
        visited.add(molecule.getId());
        Molecule copy = new Molecule(molecule);
        structure.addMolecule(copy);
        for (String bond : molecule.getBonds()) {
            Molecule nextMolecule = molecules.stream().filter(m -> m.getId().equals(bond)).findFirst().orElse(null);
            if (nextMolecule != null && !visited.contains(nextMolecule.getId())) {
                dfs(nextMolecule, visited, structure);
            }
        }
        // visit nodes if they have bond to the current molecule
        for (Molecule m : molecules) {
            if (m.getBonds().contains(molecule.getId()) && !visited.contains(m.getId())) {
                dfs(m, visited, structure);
            }
        }
    }
    public List<MolecularStructure> identifyMolecularStructures() {
        ArrayList<MolecularStructure> structures = new ArrayList<>();
        // Create a visited set to keep track of visited molecules
        HashSet<String> visited = new HashSet<>();
        // use dfs on every molecule and add the resulting structure to the list
        for (Molecule molecule : molecules) {
            if (!visited.contains(molecule.getId())) {
                MolecularStructure structure = new MolecularStructure();
                dfs(molecule, visited, structure);
                structures.add(structure);
            }
        }
        return structures;
    }

    // Method to print given molecular structures
    public void printMolecularStructures(List<MolecularStructure> molecularStructures, String species) {
        // x molecular structures have been discovered in regular humans.
        System.out.println(molecularStructures.size() + " molecular structures have been discovered in " + species + ".");
        // Molecules in Molecular Structure 1: [M46, M88, M145]
        for (int i = 0; i < molecularStructures.size(); i++) {
            System.out.println("Molecules in Molecular Structure " + (i + 1) + ": " + molecularStructures.get(i).toString());
        }
    }

    // Method to identify anomalies given a source and target molecular structure
    // Returns a list of molecular structures unique to the targetStructure only
    public static ArrayList<MolecularStructure> getVitalesAnomaly(List<MolecularStructure> sourceStructures, List<MolecularStructure> targeStructures) {
        ArrayList<MolecularStructure> anomalyList = new ArrayList<>();
        


        // Iterate through target structures and add unique structures to the anomaly list
        for (MolecularStructure targetStructure : targeStructures) {
            boolean isAnomaly = true;
            for (MolecularStructure sourceStructure : sourceStructures) {
                if (sourceStructure.equals(targetStructure)) {
                    isAnomaly = false;
                    break;
                }
            }
            if (isAnomaly) {
                anomalyList.add(new MolecularStructure(targetStructure));
            }
        }

        return anomalyList;
    }
    // Method to print Vitales anomalies
    public void printVitalesAnomaly(List<MolecularStructure> molecularStructures) {
        // Molecular structures unique to Vitales individuals:
        System.out.println("Molecular structures unique to Vitales individuals:");
        //[M13, M14, M15]
        for (MolecularStructure structure : molecularStructures) {
            System.out.println(structure.toString());
        }

    }
}
