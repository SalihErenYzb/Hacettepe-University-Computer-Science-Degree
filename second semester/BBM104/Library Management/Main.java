import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;


class FileInput {
    /**
     * Reads the file at the given path and returns contents of it in a string array.
     *
     * @param path              Path to the file that is going to be read.
     * @param discardEmptyLines If true, discards empty lines with respect to trim; else, it takes all the lines from the file.
     * @param trim              Trim status; if true, trims (strip in Python) each line; else, it leaves each line as-is.
     * @return Contents of the file as a string array, returns null if there is not such a file or this program does not have sufficient permissions to read that file.
     */
    public static String[] readFile(String path, boolean discardEmptyLines, boolean trim) {
        try {
            List<String> lines = Files.readAllLines(Paths.get(path)); //Gets the content of file to the list.
            if (discardEmptyLines) { //Removes the lines that are empty with respect to trim.
                lines.removeIf(line -> line.trim().equals(""));
            }
            if (trim) { //Trims each line.
                lines.replaceAll(String::trim);
            }
            return lines.toArray(new String[0]);
        } catch (IOException e) { //Returns null if there is no such a file.
            e.printStackTrace();
            return null;
        }
    }
}

public class Main {
	public static String opt;
    public static void main(String[] args) {
        Library library = new Library();
        opt = args[1];
        String[] lines = FileInput.readFile(args[0], true, true);
        // deleting the already existing text from output
        FileOutput.writeToFile(Main.opt,"",false,false);

        for (String line : lines) {
        	try {
            String[] parts = line.split("\t");
            String command = parts[0];

            switch (command) {
                case "addBook":

                    library.addBook( parts);
                    break;
                case "addMember":

                    library.addMember(parts);
                    break;
                case "borrowBook":

                    library.borrowBook(parts);
                    break;
                case "readInLibrary":

                    library.readInLibrary(parts);
                    break;
                case "extendBook":

                	 library.extendBook(parts);
                     break;
                case "returnBook":

                    library.returnBook(parts);
                    break;
                case "getTheHistory":

                    library.getTheHistory(parts);
                    break;
                default:

                    break;
            }
        	}catch (Exception e) {
        		FileOutput.writeToFile(opt,e.getMessage(),true,true);
        	}
        }
    }
}