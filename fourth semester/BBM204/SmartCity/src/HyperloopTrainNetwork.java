import java.io.Serializable;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.nio.file.Files;
import java.nio.file.Paths;
public class HyperloopTrainNetwork implements Serializable {
    static final long serialVersionUID = 11L;
    public double averageTrainSpeed;
    public final double averageWalkingSpeed = 1000 / 6.0;;
    public int numTrainLines;
    public Station startPoint;
    public Station destinationPoint;
    public List<TrainLine> lines;

    /**
     * Method with a Regular Expression to extract integer numbers from the fileContent
     * @return the result as int
     */
    public int getIntVar(String varName, String fileContent) {
        Pattern p = Pattern.compile("[\\t ]*" + varName + "[\\t ]*=[\\t ]*([0-9]+)");
        Matcher m = p.matcher(fileContent);
        m.find();
        return Integer.parseInt(m.group(1));
    }

    /**
     * Write the necessary Regular Expression to extract string constants from the fileContent
     * @return the result as String
     */
    public String getStringVar(String varName, String fileContent) {
        Pattern p = Pattern.compile("[\\t ]*" + varName + "[\\t ]*=[\\t ]*\"([^\"]+)\"");
        Matcher m = p.matcher(fileContent);
        m.find();
        return m.group(1);
    }

    /**
     * Write the necessary Regular Expression to extract floating point numbers from the fileContent
     * Your regular expression should support floating point numbers with an arbitrary number of
     * decimals or without any (e.g. 5, 5.2, 5.02, 5.0002, etc.).
     * @return the result as Double
     */
    public Double getDoubleVar(String varName, String fileContent) {
        Pattern p = Pattern.compile("[\\t ]*" + varName + "[\\t ]*=[\\t ]*([0-9]+(?:\\.[0-9]+)?)");
        Matcher m = p.matcher(fileContent);
        m.find();
        return Double.parseDouble(m.group(1));
    }

    /**
     * Write the necessary Regular Expression to extract a Point object from the fileContent
     * points are given as an x and y coordinate pair surrounded by parentheses and separated by a comma
     * @return the result as a Point object
     */
    public Point getPointVar(String varName, String fileContent) {
        Point p = new Point(0, 0);
        // starting_point= (0, 0 )
        Pattern pattern = Pattern.compile("[\\t ]*" + varName + "[\\t ]*=[\\t ]*\\([\\t ]*([0-9]+)[\\t ]*,[\\t ]*([0-9]+)[\\t ]*\\)");
        Matcher matcher = pattern.matcher(fileContent);
        matcher.find();
        p.x = Integer.parseInt(matcher.group(1));
        p.y = Integer.parseInt(matcher.group(2));
        return p;
    } 

    /**
     * Function to extract the train lines from the fileContent by reading train line names and their 
     * respective stations.
     * @return List of TrainLine instances
     */
    public List<TrainLine> getTrainLines(String fileContent) {
        List<TrainLine> trainLines = new ArrayList<>();
        Pattern trainLinePattern = Pattern.compile("train_line_name\\s*=\\s*\"([^\"]+)\"\\s*train_line_stations\\s*=([^\\n]*)");
        Matcher matcher = trainLinePattern.matcher(fileContent);
        
        while (matcher.find()) {
            int i = 1;
            String trainLineName = matcher.group(1);
            String stationsString = matcher.group(2);
            List<Station> stations = new ArrayList<>();
            Pattern stationPattern = Pattern.compile("\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)");
            Matcher stationMatcher = stationPattern.matcher(stationsString);

            while (stationMatcher.find()) {
                int x = Integer.parseInt(stationMatcher.group(1));
                int y = Integer.parseInt(stationMatcher.group(2));
                stations.add(new Station(new Point(x, y),trainLineName+" Line Station "+i));
                i++;
            }

            trainLines.add(new TrainLine(trainLineName, stations));
        }

        return trainLines;
    }

    /**
     * Function to populate the given instance variables of this class by calling the functions above.
     */
    public void readInput(String filename) {

        // turn the file to string
        String fileContent = new String();
        try {
            fileContent = new String(Files.readAllBytes(Paths.get(filename)));
        } catch (Exception e) {
            e.printStackTrace();
        }
        startPoint = new Station(getPointVar("starting_point", fileContent), "Starting Point");
        destinationPoint = new Station(getPointVar("destination_point", fileContent), "Final Destination");
        averageTrainSpeed = getDoubleVar("average_train_speed", fileContent)* 1000 / 60;
        numTrainLines = getIntVar("num_train_lines", fileContent);
        lines = getTrainLines(fileContent);
    }
    private void printAll(){
        // print everything
        printStation(startPoint);
        printStation(destinationPoint);
        System.out.println("Average Train Speed: " + averageTrainSpeed);
        System.out.println("Number of Train Lines: " + numTrainLines);
        for (TrainLine line : lines) {
            System.out.println("Train Line Name: " + line.trainLineName);
            for (Station station : line.trainLineStations) {
                printStation(station);
            }
        }
    }
    private static void printStation(Station station) {
        System.out.println("Station: " + station.description + " station coordinates: x: " + station.coordinates.x + " y: " + station.coordinates.y);
    }
}