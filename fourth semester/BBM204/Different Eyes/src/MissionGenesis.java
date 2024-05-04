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
    private static MolecularData processMolecularData(Document document, String tag) {
        ArrayList<Molecule> molecularData = new ArrayList<>();
        NodeList nodeList = ((Element) document.getElementsByTagName(tag).item(0)).getElementsByTagName("Molecule");
        for (int i = 0; i < nodeList.getLength(); i++) {
            Element element = (Element) nodeList.item(i);
            String id = element.getElementsByTagName("ID").item(0).getTextContent();
            String bondStrength = element.getElementsByTagName("BondStrength").item(0).getTextContent();
            NodeList bondsList = element.getElementsByTagName("MoleculeID");
            List<String> bonds = new ArrayList<>();
            for (int j = 0; j < bondsList.getLength(); j++) {
                bonds.add(bondsList.item(j).getTextContent());
            }
            molecularData.add(new Molecule(id, Integer.parseInt(bondStrength), bonds));
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
            molecularDataHuman = processMolecularData(document, "HumanMolecularData");
            molecularDataVitales = processMolecularData(document, "VitalesMolecularData");
            
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        
    }
}
