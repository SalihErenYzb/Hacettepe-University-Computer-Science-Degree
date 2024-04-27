import org.w3c.dom.*;
import javax.xml.parsers.*;
import java.io.*;
import java.util.*;
public class MissionGenesis {

    // Private fields
    private MolecularData molecularDataHuman; // Molecular data for humans
    private MolecularData molecularDataVitales; // Molecular data for Vitales

    // Getter for human molecular data
    public MolecularData getMolecularDataHuman() {
        return molecularDataHuman;
    }

    // Getter for Vitales molecular data
    public MolecularData getMolecularDataVitales() {
        return molecularDataVitales;
    }

    // Method to read XML data from the specified filename
    // This method should populate molecularDataHuman and molecularDataVitales fields once called
    private static MolecularData processMolecularData(Document document, String tag)  {
        ArrayList<Molecule> molecularData = new ArrayList<>();
        // get the tag 
        NodeList nodeListTag = document.getElementsByTagName(tag);
        Element element2 = (Element) nodeListTag.item(0);
        // Normalizing the XML Structure
        document.getDocumentElement().normalize();
        
        // Getting all the Molecule nodes in the XML
        NodeList nodeList = element2.getElementsByTagName("Molecule");
        
        for (int i = 0; i < nodeList.getLength(); i++) {
            Node node = nodeList.item(i);
            if (node.getNodeType() == Node.ELEMENT_NODE) {
                Element element = (Element) node;
                
                // Getting the ID and BondStrength of each Molecule
                String id = element.getElementsByTagName("ID").item(0).getTextContent();
                String bondStrength = element.getElementsByTagName("BondStrength").item(0).getTextContent();
                                    
                // Getting all connected MoleculeIDs
                List<String> bonds = new ArrayList<>();
                NodeList bondsList = element.getElementsByTagName("MoleculeID");
                for (int j = 0; j < bondsList.getLength(); j++) {
                    String bond = bondsList.item(j).getTextContent();
                    bonds.add(bond);
                    // add bond in the other direction
                    // Molecule otherMolecule = molecularData.stream().filter(m -> m.getId().equals(bond)).findFirst().orElse(null);
                    // if (otherMolecule != null) {
                    //     // if not in the bonds list, add it
                    //     if (!otherMolecule.getBonds().contains(id)) {
                    //         otherMolecule.getBonds().add(id);
                    //     }
                    // }
                }
                molecularData.add(new Molecule(id, Integer.parseInt(bondStrength), bonds));
            }
        }

        return new MolecularData(molecularData);

        
    }
    public void readXML(String filename) {
        try {
            // Creating a DocumentBuilderFactory
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            DocumentBuilder builder = factory.newDocumentBuilder();
            
            // Building a Document from the XML file
            Document document = builder.parse(new File(filename));
            
            // Normalizing the XML Structure
            document.getDocumentElement().normalize();
            
            // Processing HumanMolecularData
            molecularDataHuman = processMolecularData(document, "HumanMolecularData");
            // Processing VitalesMolecularData
            molecularDataVitales = processMolecularData(document, "VitalesMolecularData");
            
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        
    }
}
