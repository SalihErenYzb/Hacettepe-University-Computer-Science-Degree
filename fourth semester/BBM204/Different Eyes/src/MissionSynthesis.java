import java.util.*;

// Class representing the Mission Synthesis
public class MissionSynthesis {

    // Private fields
    private final List<MolecularStructure> humanStructures; // Molecular structures for humans
    private final ArrayList<MolecularStructure> diffStructures; // Anomalies in Vitales structures compared to humans

    // Constructor
    public MissionSynthesis(List<MolecularStructure> humanStructures, ArrayList<MolecularStructure> diffStructures) {
        this.humanStructures = humanStructures;
        this.diffStructures = diffStructures;
    }
    private HashMap<String,String> parent = new HashMap<>();
    private HashMap<String,Integer> size = new HashMap<>();
    private void addToSet(String id){
        parent.put(id,id);
        size.put(id,1);
    }
    private String find(String id){
        if(parent.get(id).equals(id)){
            return id;
        }
        parent.put(id,find(parent.get(id)));
        return parent.get(id);
    }
    private void union(String id1, String id2){
        String p1 = find(id1);
        String p2 = find(id2);
        if(p1.equals(p2)){
            return;
        }
        if(size.get(p1) < size.get(p2)){
            parent.put(p1,p2);
            size.put(p2,size.get(p1)+size.get(p2));
        }else{
            parent.put(p2,p1);
            size.put(p1,size.get(p1)+size.get(p2));
        }
    }
    // Method to synthesize bonds for the serum
    public List<Bond> synthesizeSerum() {
        List<Bond> serum = new ArrayList<>();

        ArrayList<Molecule> minHumans = new ArrayList<>();
        ArrayList<Molecule> minVitales = new ArrayList<>();
        for (MolecularStructure structure : humanStructures) {
            Molecule min = structure.getMoleculeWithWeakestBondStrength();
            minHumans.add(min);
            addToSet(min.getId());
        }
        for (MolecularStructure structure : diffStructures) {
            Molecule min = structure.getMoleculeWithWeakestBondStrength();
            minVitales.add(min);
            addToSet(min.getId());
        }
        //Typical human molecules selected for synthesis: [M145, M49, M292]
        // Vitales molecules selected for synthesis: [M13, M12]
        // Synthesizing the serum...
        System.out.println("Typical human molecules selected for synthesis: " + minHumans);
        System.out.println("Vitales molecules selected for synthesis: " + minVitales);
        System.out.println("Synthesizing the serum...");
        // Forming a bond between M13 - M145 with strength 21.00
        // Forming a bond between M49 - M145 with strength 22.00
        // Forming a bond between M145 - M292 with strength 24.00
        // Forming a bond between M12 - M145 with strength 24.00
        // finds bonds in minHUmans itself
        for (int i = 0; i < minHumans.size(); i++) {
            for (int j = i + 1; j < minHumans.size(); j++) {
                double weight = (double) (minHumans.get(i).getBondStrength() + minHumans.get(j).getBondStrength());
                serum.add(new Bond(minHumans.get(i), minHumans.get(j), weight/2.0));
                // System.out.println("Forming za bond between " + minHumans.get(i).getId() + " - " + minHumans.get(j).getId() + " with strength " + String.format("%.2f",weight/2.0));
            }
            Molecule human = minHumans.get(i);
            for (Molecule vitale : minVitales){
                double weight = (double) (human.getBondStrength() + vitale.getBondStrength());
                serum.add(new Bond(human, vitale, weight/2.0));
                // System.out.println("Forming aa bond between " + human.getId() + " - " + vitale.getId() + " with strength " + String.format("%.2f",weight/2.0));
            }
        }

        // finds bonds between minVitales itself
        for (int i = 0; i < minVitales.size(); i++) {
            for (int j = i + 1; j < minVitales.size(); j++) {
                double weight = (double) (minVitales.get(i).getBondStrength() + minVitales.get(j).getBondStrength());
                serum.add(new Bond(minVitales.get(i), minVitales.get(j), weight/2.0));
                // System.out.println("Forming ad bond between " + minVitales.get(i).getId() + " - " + minVitales.get(j).getId() + " with strength " + String.format("%.2f",weight/2.0));
            }
        }
        Collections.sort(serum, new Comparator<Bond>() {
            @Override
            public int compare(Bond o1, Bond o2) {
                return o1.getWeight().compareTo(o2.getWeight());
            }
        });
        List<Bond> serum2 = new ArrayList<>();
        for (Bond bond : serum){
            if(find(bond.getTo().getId()).equals(find(bond.getFrom().getId()))){
                continue;
            }
            union(bond.getTo().getId(),bond.getFrom().getId());
            serum2.add(bond);
        }

        return serum2;
    }

    // Method to print the synthesized bonds
    public void printSynthesis(List<Bond> serum) {
        // Forming a bond between M13 - M145 with strength 21.00
        // Forming a bond between M49 - M145 with strength 22.00
        // Forming a bond between M145 - M292 with strength 24.00
        // Forming a bond between M12 - M145 with strength 24.00
        // use MST to find the minimum spanning tree


        // The total serum bond strength is 91.00
        double total = 0.0;
        for (Bond bond : serum) {
            total += bond.getWeight();
            System.out.println("Forming a bond between " + bond.getTo().getId() + " - " + bond.getFrom().getId() + " with strength " + String.format("%.2f",bond.getWeight()));
        }
        System.out.println("The total serum bond strength is " + String.format("%.2f",total));

        

    }
}
