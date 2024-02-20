import java.util.*;


import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.time.format.DateTimeParseException;

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

class FileOutput {
	 /**
	  * This function writes given content to file at given path.
	  *
	  * @param path    Path for the file content is going to be written.
	  * @param content Content that is going to be written to file.
	  * @param append  Append status, true if wanted to append to file if it exists, false if wanted to create file from zero.
	  * @param newLine True if wanted to append a new line after content, false if vice versa.
	  */
	 
	 public static void writeToFile(String path, String content, boolean append, boolean newLine) {
	     PrintStream ps = null;
	     try {
	         ps = new PrintStream(new FileOutputStream(path, append));
	         ps.print(content + (newLine ? "\n" : ""));
	     } catch (FileNotFoundException e) {
	         e.printStackTrace();
	     } finally {
	         if (ps != null) { //Flushes all the content and closes the stream if it has been successfully created.
	             ps.flush();
	             ps.close();
	         }
	     }
	 }
	}



public class Main {
    public static List<SmartDevice> devices = new ArrayList<>();
    static LocalDateTime time;
    public static String output;


    public static String[] splitStringByTab(String input) {
        // Split the input string by tabs
        String[] parts = input.split("\\t");
        
        // Return the array of parts
        return parts;
    }
    /**

    This method is the entry point of the program.

    It reads the input file and processes the commands.

    It also generates an output file that contains the result of each command.

    @param args Command-line arguments passed to the program.
    */
    public static void main(String[] args) {
    	int deneme = 0;
        // Read input from file and store in an array of strings
    	output = args[1];
    	String[] input = FileInput.readFile(args[0],true,true); 
        // Initialize a boolean variable to control the loop

    	boolean terminate = false;
        // Loop through each line of input

    	for (String line : input) {
            // If terminate is true, break out of the loop

    		if (terminate) {break;}
            // Increment the counter

    		deneme++;
            // Split the line by tab and store in an array of strings

    		String[] tmpParts = splitStringByTab(line);
            // Get the function name from the first element of the array

    		String function = tmpParts[0];
            // Write the command to the output file

    		FileOutput.writeToFile(Main.output, "COMMAND: "+line,true,true);
            // If this is the first command and it's not SetInitialTime, terminate the program
    		if (deneme == 1 && !function.equals("SetInitialTime") ){
    			FileOutput.writeToFile(Main.output, "ERROR: First command must be set initial time! Program is going to terminate!", true, true);
    			break;
    			
    		}
    	    try {
                // Use a switch statement to execute the appropriate function based on the function name
    	        switch (function) {
    	        	case "SetInitialTime":
    	            	// Set the initial time
                		// If this is the first command

    	        		if (deneme==1) {
    	        			// If there are more than two arguments or no arguments, terminate the program
    	        			if (tmpParts.length>2 ) {
    	        				terminate = true;
    	        				throw new IllegalArgumentException("ERROR: Erroneous command!");
    	        			}else if(tmpParts.length==1) {
    	        				terminate = true;
    	        				throw new IllegalArgumentException("ERROR: First command must be set initial time! Program is going to terminate!");
    	        			}
    	        			// If the format of the initial time is wrong, terminate the program
    	        			try {
    	        				SetTime(tmpParts);
    	        			}catch(Exception e) {
    	            			terminate = true;
    	            			throw new IllegalArgumentException("ERROR: Format of the initial date is wrong! Program is going to terminate!");
    	        			}
                			// Write a success message to the output file

    	                FileOutput.writeToFile(Main.output,"SUCCESS: Time has been set to "+tmpParts[1]+"!",true,true);
            			// If it's not the first command, write an error message to the output file

    	        		}else {
    	                FileOutput.writeToFile(Main.output,"ERROR: Erroneous command!",true,true);
    	        		}
    	                break;	
    	            case "SetTime":
    	                SetTime(tmpParts);
    	                break;
    	            case "SkipMinutes":
    	                SkipMinutes(tmpParts);
    	                break;
    	            case "Nop":
    	                Nop();
    	                break;
    	            case "Add":
    	                Add(tmpParts);
    	                break;
    	            case "Remove":
    	                Remove(tmpParts);
    	                break;
    	            case "SetSwitchTime":
    	                SetSwitchTime(tmpParts);
    	                break;
    	            case "Switch":
    	                Switch(tmpParts);
    	                break;
    	            case "ChangeName":
    	                ChangeName(tmpParts);
    	                break;
    	            case "PlugIn":
    	                PlugIn(tmpParts);
    	                break;
    	            case "PlugOut":
    	                PlugOut(tmpParts);
    	                break;
    	            case "SetKelvin":
    	                SetKelvin(tmpParts);
    	                break;
    	            case "SetBrightness":
    	                SetBrightness(tmpParts);
    	                break;
    	            case "SetColorCode":
    	                SetColorCode(tmpParts);
    	                break;
    	            case "SetWhite":
    	                SetWhite(tmpParts);
    	                break;
    	            case "SetColor":
    	                SetColor(tmpParts);
    	                break;
    	            case "ZReport":
    	            	ZReport();
    	            	break;
    	            default:
    	                // If the function name is invalid, write an error message to the output file

    	                FileOutput.writeToFile(Main.output, "ERROR: Erroneous command!", true, true);
    	                break;
    	        }
    	    } catch (Exception e) {
                // Write the error message to the output file

    	    	FileOutput.writeToFile(Main.output, e.getMessage(), true, true);
    	    }
    	}
    }
    /**
     * This function prints the current time and the status of all SmartDevice objects in the devices list.
     */
    private static void ZReport() {
        // Create a DateTimeFormatter object with the desired format
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss");

        // Format the current time using the formatter object and store it in a string variable
        String formattedDateTime = Main.time.format(formatter);

        // Write the formatted date and time to the output file
        FileOutput.writeToFile(Main.output, String.format("Time is:\t%s", formattedDateTime), true, true);

        // Loop through each device in the devices list and print its status
        for (SmartDevice device : devices) {
            device.printStatus();
        }
    }
    /**
     * This function sets the color and brightness of a SmartColorLamp object with the given name.
     * @param tmpParts An array of strings containing the command arguments.
     *                 tmpParts[1] is the name of the device.
     *                 tmpParts[2] is the color code.
     *                 tmpParts[3] is the brightness level.
     */
    private static void SetColor(String[] tmpParts) {
        // Initialize a boolean variable to keep track of whether the device was found
        boolean found = false;

        // Get an iterator for the devices list
        Iterator<SmartDevice> it = devices.iterator();

        // Declare a variable to hold the SmartColorLamp object
        SmartColorLamp device;

        // Loop through each device in the devices list
        while (it.hasNext()) {
            // Get the next device in the list
            SmartDevice tmpDvc = it.next();

            // If the name of the device matches the name passed in the function argument
            if (tmpDvc.getName().equals(tmpParts[1])) {
                try {
                    // Set found to true
                    found = true;

                    // Cast the device to a SmartColorLamp object
                    device = (SmartColorLamp)tmpDvc;

                    // Store the old brightness level
                    int oldValue = device.brightness;

                    // Set the brightness level of the device
                    device.setBrightness(tmpParts[3]);

                    // Set the color code of the device
                    try {
                        device.setColorCode(tmpParts[2]);
                    }
                    // If the color code is invalid, restore the old brightness level and throw an exception
                    catch(IllegalArgumentException e) {
                        device.brightness = oldValue;
                        throw new IllegalArgumentException("ERROR: Erroneous command!");
                    }

                    // Exit the loop since the device has been found and updated
                    break;
                }
                // If the device is not a SmartColorLamp, throw an exception
                catch(ClassCastException e) {
                    throw new IllegalArgumentException("ERROR: This device is not a smart color lamp!");
                }
            }
        }

        // If the device was not found, write an error message to the output file
        if (!found) {
            FileOutput.writeToFile(Main.output, "ERROR: The device " + tmpParts[1] + " was not found.", true, true);
        }					
    }
    /**
     * This function sets the color temperature and brightness of a SmartLamp object with the given name.
     * @param tmpParts An array of strings containing the command arguments.
     *                 tmpParts[1] is the name of the device.
     *                 tmpParts[2] is the color temperature in kelvin.
     *                 tmpParts[3] is the brightness level.
     */
    private static void SetWhite(String[] tmpParts) {
        // Initialize a boolean variable to keep track of whether the device was found
        boolean found = false;

        // Get an iterator for the devices list
        Iterator<SmartDevice> it = devices.iterator();

        // Declare a variable to hold the SmartLamp object
        SmartLamp device;

        // Loop through each device in the devices list
        while (it.hasNext()) {
            // Get the next device in the list
            SmartDevice tmpDvc = it.next();

            // If the name of the device matches the name passed in the function argument
            if (tmpDvc.getName().equals(tmpParts[1])) {
                try {
                    // Cast the device to a SmartLamp object
                    device = (SmartLamp)tmpDvc;

                    // Set found to true
                    found = true;

                    // Set the color temperature and brightness level of the device
                    device.setKelvin(tmpParts[2]);
                    device.setBrightness(tmpParts[3]);

                    // Exit the loop since the device has been found and updated
                    break;
                }
                // If the device is not a SmartLamp, throw an exception
                catch(ClassCastException e) {
                    throw new IllegalArgumentException("ERROR: This device is not a smart lamp!");
                }
            }
        }

        // If the device was not found, write an error message to the output file
        if (!found) {
            FileOutput.writeToFile(Main.output, "ERROR: The device " + tmpParts[1] + " was not found.", true, true);
        }			
    }
    /**
     * This function sets the color code of a SmartColorLamp object with the given name.
     * @param tmpParts An array of strings containing the command arguments.
     *                 tmpParts[1] is the name of the device.
     *                 tmpParts[2] is the color code to set.
     */
    private static void SetColorCode(String[] tmpParts) {
        // Initialize a boolean variable to keep track of whether the device was found
        boolean found = false;

        // Get an iterator for the devices list
        Iterator<SmartDevice> it = devices.iterator();

        // Declare a variable to hold the SmartColorLamp object
        SmartColorLamp device;

        // Loop through each device in the devices list
        while (it.hasNext()) {
            // Get the next device in the list
            SmartDevice tmpDvc = it.next();

            // If the name of the device matches the name passed in the function argument
            if (tmpDvc.getName().equals(tmpParts[1])) {
                try {
                    // Cast the device to a SmartColorLamp object
                    device = (SmartColorLamp)tmpDvc;

                    // Set found to true
                    found = true;

                    // Set the color code of the device
                    device.setColorCode(tmpParts[2]);

                    // Exit the loop since the device has been found and updated
                    break;
                }
                // If the device is not a SmartColorLamp, throw an exception
                catch(ClassCastException e) {
                    throw new IllegalArgumentException("ERROR: This device is not a smart color lamp!");
                }
            }
        }

        // If the device was not found, write an error message to the output file
        if (!found) {
            FileOutput.writeToFile(Main.output, "ERROR: The device " + tmpParts[1] + " was not found.", true, true);
        }					
    }
    /**
     * This function sets the brightness level of a SmartLamp object with the given name.
     * @param tmpParts An array of strings containing the command arguments.
     *                 tmpParts[1] is the name of the device.
     *                 tmpParts[2] is the brightness level to set.
     */
    private static void SetBrightness(String[] tmpParts) {
        // Initialize a boolean variable to keep track of whether the device was found
        boolean found = false;

        // Get an iterator for the devices list
        Iterator<SmartDevice> it = devices.iterator();

        // Declare a variable to hold the SmartLamp object
        SmartLamp device;

        // Loop through each device in the devices list
        while (it.hasNext()) {
            // Get the next device in the list
            SmartDevice tmpDvc = it.next();

            // If the name of the device matches the name passed in the function argument
            if (tmpDvc.getName().equals(tmpParts[1])) {
                try {
                    // Cast the device to a SmartLamp object
                    device = (SmartLamp)tmpDvc;

                    // Set found to true
                    found = true;

                    // Set the brightness level of the device
                    device.setBrightness(tmpParts[2]);

                    // Exit the loop since the device has been found and updated
                    break;
                }
                // If the device is not a SmartLamp, throw an exception
                catch(ClassCastException e) {
                    throw new IllegalArgumentException("ERROR: This device is not a smart lamp!");
                }

            }
        }

        // If the device was not found, write an error message to the output file
        if (!found) {
            FileOutput.writeToFile(Main.output, "ERROR: The device " + tmpParts[1] + " was not found.", true, true);
        }				
    }
    /**
     * This function sets the color temperature (in Kelvin) of a SmartLamp object with the given name.
     * @param tmpParts An array of strings containing the command arguments.
     *                 tmpParts[1] is the name of the device.
     *                 tmpParts[2] is the color temperature (in Kelvin) to set.
     */
    private static void SetKelvin(String[] tmpParts) {
        // Initialize a boolean variable to keep track of whether the device was found
        boolean found = false;

        // Get an iterator for the devices list
        Iterator<SmartDevice> it = devices.iterator();

        // Declare a variable to hold the SmartLamp object
        SmartLamp device;

        // Loop through each device in the devices list
        while (it.hasNext()) {
            // Get the next device in the list
            SmartDevice tmpDvc = it.next();

            // If the name of the device matches the name passed in the function argument
            if (tmpDvc.getName().equals(tmpParts[1])) {
                try {
                    // Cast the device to a SmartLamp object
                    device = (SmartLamp)tmpDvc;

                    // Set found to true
                    found = true;

                    // Set the color temperature (in Kelvin) of the device
                    device.setKelvin(tmpParts[2]);

                    // Exit the loop since the device has been found and updated
                    break;
                }
                // If the device is not a SmartLamp, throw an exception
                catch(ClassCastException e) {
                    throw new IllegalArgumentException("ERROR: This device is not a smart lamp!");
                }

            }
        }

        // If the device was not found, write an error message to the output file
        if (!found) {
            FileOutput.writeToFile(Main.output, "ERROR: The device " + tmpParts[1] + " was not found.", true, true);
        }				
    }
    /**
     * This function unplugs a SmartPlug object with the given name.
     * @param tmpParts An array of strings containing the command arguments.
     *                 tmpParts[1] is the name of the device.
     */
    private static void PlugOut(String[] tmpParts) {
        // Initialize a boolean variable to keep track of whether the device was found
        boolean found = false;

        // Get an iterator for the devices list
        Iterator<SmartDevice> it = devices.iterator();

        // Declare a variable to hold the SmartPlug object
        SmartPlug device;

        // Loop through each device in the devices list
        while (it.hasNext()) {
            // Get the next device in the list
            SmartDevice tmpDvc = it.next();

            // If the name of the device matches the name passed in the function argument
            if (tmpDvc.getName().equals(tmpParts[1])) {
                // Set found to true
                found = true;

                // If the device is a SmartPlug, unplug it
                if (tmpDvc instanceof SmartPlug) {
                    // Cast the device to a SmartPlug object
                    device = (SmartPlug)tmpDvc;

                    // Unplug the device
                    device.unplug();
                }
                // If the device is not a SmartPlug, write an error message to the output file
                else {
                    FileOutput.writeToFile(Main.output, "ERROR: The device " + tmpParts[1] + " is not a SmartPlug.", true, true);
                }

                // Exit the loop since the device has been found and unplugged
                break;
            }
        }

        // If the device was not found, write an error message to the output file
        if (!found) {
            FileOutput.writeToFile(Main.output, "ERROR: The device " + tmpParts[1] + " was not found.", true, true);
        }				
    }
    /**
     * This function plugs in a SmartPlug object with the given name and sets its amperage.
     * @param tmpParts An array of strings containing the command arguments.
     *                 tmpParts[1] is the name of the device.
     *                 tmpParts[2] is the amperage to set.
     */
    private static void PlugIn(String[] tmpParts) {
        // Initialize a boolean variable to keep track of whether the device was found
        boolean found = false;

        // Get an iterator for the devices list
        Iterator<SmartDevice> it = devices.iterator();

        // Declare a variable to hold the SmartPlug object
        SmartPlug device;

        // Loop through each device in the devices list
        while (it.hasNext()) {
            // Get the next device in the list
            SmartDevice tmpDvc = it.next();

            // If the name of the device matches the name passed in the function argument
            if (tmpDvc.getName().equals(tmpParts[1])) {
                // Set found to true
                found = true;

                // If the device is a SmartPlug, set its amperage
                if (tmpDvc instanceof SmartPlug) {
                    // Cast the device to a SmartPlug object
                    device = (SmartPlug)tmpDvc;

                    // Set the amperage of the device
                    device.setAmpere(tmpParts[2]);
                }
                // If the device is not a SmartPlug, write an error message to the output file
                else {
                    FileOutput.writeToFile(Main.output, "ERROR: This device is not a smart plug!", true, true);
                }

                // Exit the loop since the device has been found and plugged in
                break;
            }
        }

        // If the device was not found, write an error message to the output file
        if (!found) {
            FileOutput.writeToFile(Main.output, "ERROR: The device " + tmpParts[1] + " was not found.", true, true);
        }		
    }
    /**
     * This function changes the name of a smart device with the given name to a new name.
     * @param tmpParts An array of strings containing the command arguments.
     *                 tmpParts[1] is the current name of the device.
     *                 tmpParts[2] is the new name to set.
     * @throws IllegalArgumentException if the current name and new name are the same or if there is already a device with the new name.
     */
    private static void ChangeName(String[] tmpParts) throws IllegalArgumentException {
        try {
            // Initialize a boolean variable to keep track of whether the device was found
            boolean found = false;

            // Get an iterator for the devices list
            Iterator<SmartDevice> it = devices.iterator();

            // Check if the current name and new name are the same
            if (tmpParts[1].equals(tmpParts[2])) {
                throw new IllegalArgumentException("ERROR: Both of the names are the same, nothing changed!");
            }

            // Loop through each device in the devices list
            while (it.hasNext()) {
                SmartDevice device = it.next();

                // If there is already a device with the new name, throw an exception
                if (device.getName().equals(tmpParts[2])) {
                    throw new IllegalArgumentException("ERROR: There is already a smart device with same name!");
                }
            }

            // Get another iterator for the devices list
            Iterator<SmartDevice> it1 = devices.iterator();

            // Loop through each device in the devices list
            while (it1.hasNext()) {
                SmartDevice device = it1.next();

                // If the device has the current name, update its name to the new name
                if (device.getName().equals(tmpParts[1])) {
                    found = true;
                    device.setName(tmpParts[2]);
                }
            }

            // If the device was not found, write an error message to the output file
            if (!found) {
                FileOutput.writeToFile(Main.output, "ERROR: The device " + tmpParts[1] + " was not found.", true, true);
            }
        }
        // If there is an array index out of bounds error, throw an exception
        catch (ArrayIndexOutOfBoundsException e) {
            throw new IllegalArgumentException("ERROR: Erroneous command!");
        }		
    }
    /**
     * This function switches a smart device with the given name on or off.
     * @param tmpParts An array of strings containing the command arguments.
     *                 tmpParts[1] is the name of the device.
     *                 tmpParts[2] is the status to set (either "on" or "off").
     * @throws IllegalArgumentException if the device is already switched on or if the device does not exist.
     */
    private static void Switch(String[] tmpParts) throws IllegalArgumentException {
        // Initialize a boolean variable to keep track of whether the device was found
        boolean found = false;

        // Get an iterator for the devices list
        Iterator<SmartDevice> it = devices.iterator();

        // Loop through each device in the devices list
        while (it.hasNext()) {
            SmartDevice device = it.next();

            // If the device has the given name, toggle its status
            if (device.getName().equals(tmpParts[1])) {
                found = true;

                // If the device is already switched on, throw an exception
                if (device.getStatus() == device.getStatus(tmpParts[2])) {
                    FileOutput.writeToFile(Main.output, "ERROR: This device is already switched on!", true, true);
                    break;
                }

                // Set the device's status to the given status
                device.setStatus(tmpParts[2]);
                break;
            }
        }

        // If the device was not found, write an error message to the output file
        if (!found) {
            FileOutput.writeToFile(Main.output, "ERROR: There is not such a device!", true, true);
        }		
    }
    /**
     * This function sets the switch time for a smart device with the given name.
     * @param tmpParts An array of strings containing the command arguments.
     *                 tmpParts[1] is the name of the device.
     *                 tmpParts[2] is the switch time to set in the format "yyyy-MM-dd_HH:mm:ss".
     * @throws IllegalArgumentException if the switch time is in the past.
     * @throws NoSuchElementException if the device with the given name does not exist in the devices list.
     */
    private static void SetSwitchTime(String[] tmpParts) throws IllegalArgumentException, NoSuchElementException {
        // Initialize a boolean variable to keep track of whether the device was found
        boolean found = false;

        // Get an iterator for the devices list
        Iterator<SmartDevice> it = devices.iterator();

        // Loop through each device in the devices list
        while (it.hasNext()) {
            SmartDevice device = it.next();

            // If the device has the given name, set its switch time
            if (device.getName().equals(tmpParts[1])) {
                found = true;

                // Parse the given switch time string into a LocalDateTime object
                LocalDateTime tmpSwitch = LocalDateTime.parse(tmpParts[2], DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss"));

                // If the device does not already have a switch time set or if the new switch time is different from the old one
                if (device.getSwitchTime() == null || !tmpSwitch.isEqual(device.getSwitchTime())) {
                    // If the switch time is in the past, throw an exception
                    if (tmpSwitch.isBefore(Main.time)) {
                        throw new IllegalArgumentException("ERROR: Switch time cannot be in the past!");
                    }
                    
                    // Remove the device from the list and set its switch time to the new switch time
                    it.remove();
                    device.setSwitchTime(tmpParts[2]);

                    break;
                }
            }
        }

        // If the device was not found, write an error message to the output file
        if (!found) {
            FileOutput.writeToFile(Main.output, "ERROR: The device " + tmpParts[1] + " was not found.", true, true);
            throw new NoSuchElementException("Device not found.");
        }
    }
    /**
     * This function removes a smart device with the given name from the devices list.
     * @param tmpParts An array of strings containing the command arguments.
     *                 tmpParts[1] is the name of the device to remove.
     * @throws NoSuchElementException if the device with the given name does not exist in the devices list.
     */
    private static void Remove(String[] tmpParts) throws NoSuchElementException {
        // Initialize a boolean variable to keep track of whether the device was found
        boolean found = false;

        // Get an iterator for the devices list
        Iterator<SmartDevice> it = devices.iterator();

        // Loop through each device in the devices list
        while (it.hasNext()) {
            SmartDevice device = it.next();

            // If the device has the given name, remove it from the list
            if (device.getName().equals(tmpParts[1])) {
                found = true;

                // If the device is currently on, turn it off
                if (device.getStatus()) {
                    device.setStatus(false);
                }

                // Write a success message to the output file and print the device's status
                FileOutput.writeToFile(Main.output, "SUCCESS: Information about removed smart device is as follows:", true, true);
                device.printStatus();

                it.remove();
                break;
            }
        }

        // If the device was not found, write an error message to the output file
        if (!found) {
            FileOutput.writeToFile(Main.output, "ERROR: The device " + tmpParts[1] + " was not found in the list.", true, true);
            throw new NoSuchElementException("Device not found.");
        }
    }
    /**
     * This function adds a smart device to the devices list in the correct order based on its switch time.
     * @param deviceTmp The SmartDevice object to add to the list.
     */
    public static void AddToList(SmartDevice deviceTmp) {
        // Initialize an index variable to keep track of where to insert the new device
        int index = 0;

        // If the device does not have a switch time, add it to the end of the list
        if (deviceTmp.getSwitchTime() == null) {
            devices.add(deviceTmp);
        } else {
            // Find the correct position to insert the new device based on its switch time
            while (index < devices.size() && (devices.get(index).getSwitchTime().isAfter(deviceTmp.getSwitchTime())
                    || devices.get(index).getSwitchTime().isEqual(deviceTmp.getSwitchTime()))) {
                index++;
            }

            // Insert the new device at the correct position
            devices.add(index, deviceTmp);
        }
    }
    /**
     * This function adds a new smart device to the devices list based on the command arguments.
     * @param tmpParts An array of strings containing the command arguments.
     *                 tmpParts[1] is the type of the device to add.
     *                 tmpParts[2] is the name of the device to add.
     *                 Additional arguments may be required depending on the device type.
     */
    private static void Add(String[] tmpParts) {
        try {
            // Check if a device with the same name already exists in the devices list
            boolean found = true;
            for (SmartDevice device : devices) {
                if (device.getName().equals(tmpParts[2])) {
                    found = false;
                    FileOutput.writeToFile(Main.output, "ERROR: There is already a smart device with same name!", true, true);
                    break;
                }
            }
            
            // If a device with the same name was not found, add the new device to the devices list
            if (found) {
                switch (tmpParts[1]) {
                    case "SmartColorLamp":
                        SmartColorLamp deviceTmp;
                        
                        // Create a new SmartColorLamp object based on the command arguments
                        if (tmpParts.length == 3) {
                            deviceTmp = new SmartColorLamp(tmpParts[2]);
                        } else if (tmpParts.length == 4) {
                            deviceTmp = new SmartColorLamp(tmpParts[2], tmpParts[3]);
                        } else if (tmpParts.length == 6) {
                            deviceTmp = new SmartColorLamp(tmpParts[2], tmpParts[3], tmpParts[4], tmpParts[5]);
                        } else {
                            throw new IllegalArgumentException("ERROR: Invalid number of parameters");
                        }
                        
                        // Add the new device to the devices list in the correct order
                        AddToList(deviceTmp);
                        break;
                        
                    case "SmartLamp":
                        SmartLamp deviceTmp2;
                        
                        // Create a new SmartLamp object based on the command arguments
                        if (tmpParts.length == 3) {
                            deviceTmp2 = new SmartLamp(tmpParts[2]);
                        } else if (tmpParts.length == 4) {
                            deviceTmp2 = new SmartLamp(tmpParts[2], tmpParts[3]);
                        } else if (tmpParts.length == 6) {
                            deviceTmp2 = new SmartLamp(tmpParts[2], tmpParts[3], tmpParts[4], tmpParts[5]);
                        } else {
                            throw new IllegalArgumentException("ERROR: Invalid number of parameters");
                        }
                        
                        // Add the new device to the devices list in the correct order
                        AddToList(deviceTmp2);
                        break;

                    case "SmartPlug":
                        SmartPlug deviceTmp3;
                        
                        // Create a new SmartPlug object based on the command arguments
                        if (tmpParts.length == 3) {
                            deviceTmp3 = new SmartPlug(tmpParts[2]);
                        } else if (tmpParts.length == 4) {
                            deviceTmp3 = new SmartPlug(tmpParts[2], tmpParts[3]);
                        } else if (tmpParts.length == 5) {
                            deviceTmp3 = new SmartPlug(tmpParts[2], tmpParts[3], tmpParts[4]);
                        } else {
                            throw new IllegalArgumentException("ERROR: Invalid number of parameters");
                        }
                        
                        // Add the new device to the devices list in the correct order
                        AddToList(deviceTmp3);
                        break;

                    case "SmartCamera":
                        SmartCamera deviceTmp4;
                        
                        // Create a new SmartCamera object based on the command arguments
                        if (tmpParts.length == 4) {
                            deviceTmp4 = new SmartCamera(tmpParts[2], tmpParts[3]);
                        } else if (tmpParts.length == 5) {
                            deviceTmp4 = new SmartCamera(tmpParts[2], tmpParts[3], tmpParts[4]);
                        } else {
                            throw new IllegalArgumentException("ERROR: Invalid number of parameters");
                        }
                        
                        // Add the new device to the devices list in the correct order
                        AddToList(deviceTmp4);
                        break;
                }
            } else {
                // If a device with the same name was found, do not add the new device
            }
        } catch (IllegalArgumentException e) {
            // If an error occurs during the creation of the new device, write an error message to the output file
            FileOutput.writeToFile(Main.output, e.getMessage(), true, true);
        }
    }
	
	
    /**
     * Switches the devices according to their switch time.
     * Throws an exception if there are no devices to switch or if no device has a switch time.
     */
    public static void Nop() {
        if (devices.isEmpty()) { // checks if the list of devices is empty
            throw new IllegalArgumentException("ERROR: There is nothing to switch!");
        }

        SmartDevice device = devices.get(0); // gets the first device in the list

        if (!device.hasSwitchTime()) { // checks if the first device has a switch time
            throw new IllegalArgumentException("ERROR: No Device Has Switch Time");
        }

        LocalDateTime switchTime = device.getSwitchTime();
        Main.time = switchTime; // sets the current time to the switch time of the first device

        // Find the index of the first occurrence of a device with a null switch time
        int index = -1;
        for (int i = 0; i < devices.size(); i++) {
            SmartDevice d = devices.get(i);
            if (d != null && !d.hasSwitchTime()) { // finds the first device with a null switch time
                index = i;
                break;
            }
        }
        if (index == -1) {
            index = devices.size(); // if no device with a null switch time was found, add the device to the end of the list
        }
        int i2 = 0;
        while (true) {
            SmartDevice d2 = devices.get(i2);
            if (d2 != null && d2.hasSwitchTime() && d2.getSwitchTime().equals(Main.time)) { // if the device has a switch time equal to the current time
                try {
                    d2.setStatus("On"); // turn the device on
                } catch (Exception e) {
                    d2.setStatus("Off"); // if there's an error, turn the device off
                }
                d2.switchTime = null; // set the device's switch time to null
                devices.add(index, d2); // add the device to the list at the index found earlier
                devices.remove(0); // remove the first device from the list

            } else {
                break; // if the device doesn't have a switch time equal to the current time, exit the loop
            }
        }
    }
    /**
     * Skips the specified number of minutes and switches the devices with switch times before or at the new time.
     * Throws an exception if the command is erroneous or if the time is negative or zero.
     *
     * @param tmpParts an array of strings containing the command and the number of minutes to skip.
     * @throws IllegalArgumentException if the command is erroneous or if the time is negative or zero.
     */
    private static void SkipMinutes(String[] tmpParts) throws IllegalArgumentException {
        int tmp;
        try {
            tmp = Integer.parseInt(tmpParts[1]); // convert the second element of the array to an integer
            if (tmpParts.length > 2) {
                throw new IllegalArgumentException("ERROR: Erroneous command!");
            }
            if (tmp < 0) {
                throw new IllegalArgumentException("ERROR: Time cannot be reversed!");
            } else if (tmp == 0) {
                throw new IllegalArgumentException("ERROR: There is nothing to skip!");
            }
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("ERROR: Erroneous command!");
        }

        LocalDateTime tmpTime = time.plusMinutes(tmp); // add the specified number of minutes to the current time
        int i = 0;
        while (true) {
            try {
                SmartDevice device = devices.get(i);
                if (device.hasSwitchTime()) {
                    if (device.getSwitchTime().isBefore(tmpTime)) { // if the device's switch time is before the new time, switch the device
                        Nop();
                    } else {
                        break; // if the device's switch time is after the new time, exit the loop
                    }
                } else {
                    break; // if the device doesn't have a switch time, exit the loop
                }
            } catch (Exception e) {
                break; // if an exception is thrown, exit the loop
            }
        }
        time = tmpTime; // set the current time to the new time
    }
    /**
     * Sets the current time to the specified time and switches the devices with switch times before or at the new time.
     * Throws an exception if the time is in the wrong format, if the time is before the current time, or if the time is the same as the current time.
     *
     * @param tmpParts an array of strings containing the command and the new time in the format "yyyy-MM-dd_HH:mm:ss".
     * @throws IllegalArgumentException if the time is in the wrong format, if the time is before the current time, or if the time is the same as the current time.
     */
    private static void SetTime(String[] tmpParts) {
        try {
            String datetimeString = tmpParts[1];
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss"); // create a formatter for the datetime string
            LocalDateTime datetime = LocalDateTime.parse(datetimeString, formatter); // parse the datetime string to a LocalDateTime object
            if (Main.time != null && datetime.isBefore(Main.time)) { // if the new time is before the current time, throw an exception
                throw new IllegalArgumentException("ERROR: Time cannot be reversed!");
            } else if (Main.time != null && datetime.equals(Main.time)) { // if the new time is the same as the current time, throw an exception
                throw new IllegalArgumentException("ERROR: There is nothing to change!");
            }
            LocalDateTime tmpTime = datetime;
            int i = 0;
            while (true) {
                try {
                    SmartDevice device = devices.get(i);
                    if (device.hasSwitchTime()) {
                        if (device.getSwitchTime().isBefore(tmpTime)) { // if the device's switch time is before the new time, switch the device
                            Nop();
                        } else {
                            i++;
                            break; // if the device's switch time is after the new time, exit the loop
                        }
                    } else {
                        i++;
                        break; // if the device doesn't have a switch time, exit the loop
                    }
                } catch (IndexOutOfBoundsException e) {
                    break; // if an index out of bounds exception is thrown, exit the loop
                }
            }
            time = datetime; // set the current time to the new time
        } catch (DateTimeParseException e) {
            throw new IllegalArgumentException("ERROR: Time format is not correct!");
        }
    }
    /**
     * Converts a hexadecimal string to an integer.
     * Throws an exception if the input string is not a valid hexadecimal string, if the hexadecimal string is longer than 6 characters, or if the input string has an erroneous command.
     *
     * @param hexString a string representing a hexadecimal value, optionally preceded with "0x".
     * @return the integer value of the hexadecimal string.
     * @throws IllegalArgumentException if the input string is not a valid hexadecimal string, if the hexadecimal string is longer than 6 characters, or if the input string has an erroneous command.
     */
    public static int hexStringToInt(String hexString) throws IllegalArgumentException {
        // Remove any leading "0x" if present
        if (hexString.startsWith("0x")) {
            hexString = hexString.substring(2);
        } else {
            throw new IllegalArgumentException("ERROR: Erroneous command!");
        }

        String trimmedString = "";
        boolean foundNonzero = false;

        // Trim leading zeros
        for (int i = 0; i < hexString.length(); i++) {
            char c = hexString.charAt(i);
            if (c != '0') {
                foundNonzero = true;
            }
            if (foundNonzero) {
                trimmedString += c;
            }
        }

        // Check if string is longer than 6 characters
        if (trimmedString.length() > 6) {
            throw new IllegalArgumentException("ERROR: Color code value must be in range of 0x0-0xFFFFFF!");
        }

        // Convert to integer
        try {
            return Integer.parseInt(trimmedString, 16);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("ERROR: Erroneous command!");
        }
    }
}