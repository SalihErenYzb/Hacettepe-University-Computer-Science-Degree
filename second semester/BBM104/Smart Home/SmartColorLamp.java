import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

// The SmartColorLamp class extends the SmartLamp class and represents a smart lamp that can emit light in different colors.
class SmartColorLamp extends SmartLamp {
    private int colorCode;
    boolean colorMode = false; // A boolean variable to keep track of whether the lamp is in color mode or not.
    
    // Constructor that takes the name of the lamp and sets it using the superclass constructor.
    public SmartColorLamp(String name) {
        super(name);
    }
    
    // Constructor that takes the name of the lamp and its status and sets them using the superclass constructor and setStatus() method.
    public SmartColorLamp(String name, String str) {
        super(name);
        this.setStatus(str);
    }
    
 // Constructor method for the SmartColorLamp class that takes four arguments: the name of the lamp, its status, its color or kelvin value, and brightness.
    public SmartColorLamp(String name, String str, String kelOrColor, String brightness) {
        // Call the superclass constructor to set the name of the lamp.
        super(name);
        // Set the status of the lamp using the setStatus() method.
        this.setStatus(str);
        // Set the brightness of the lamp using the setBrightness() method.
        this.setBrightness(brightness);
        
        // Check if the kelOrColor argument contains the letter "x" (indicating that the value is a color code) using the contains() method.
        if (kelOrColor.contains("x")) {
            // If it does contain "x", call the setColorCode() method to set the color code of the lamp.
            this.setColorCode(kelOrColor);
        } else {
            // If it doesn't contain "x", assume that the value is a kelvin value and call the setKelvin() method to set the kelvin value of the lamp.
            this.setKelvin(kelOrColor);
        }
    }
    
    /**
     * Sets the kelvin value of the lamp.
     * @param kelvin A string representing the kelvin value to set.
     * @throws IllegalArgumentException If the kelvin value is outside the valid range (2000K-6500K) or the input is not a number.
     */
    public void setKelvin(String kelvin) throws IllegalArgumentException {
        try {
            // Convert the kelvin string input to an integer.
            int kelvin1 = Integer.parseInt(kelvin);
            // Check if the kelvin value is within the valid range (2000K-6500K).
            if (kelvin1 >= 2000 && kelvin1 <= 6500) {
                // If the kelvin value is valid, set the kelvin property of the lamp and set the color mode to false.
                this.kelvin = kelvin1;
                colorMode = false;
            } else {
                // If the kelvin value is not within the valid range, throw an IllegalArgumentException with an error message.
                throw new IllegalArgumentException("ERROR: Kelvin value must be in range of 2000K-6500K!");
            }
        } catch (NumberFormatException e) {
            // If the input is not a number, throw an IllegalArgumentException with an error message.
            throw new IllegalArgumentException("ERROR: Erroneous command!");
        }
    }
    // Method to set the color code of the lamp and switch to color mode.
    public void setColorCode(String colorCode) {
        this.colorCode = Main.hexStringToInt(colorCode);
        colorMode = true;
    }
    
    /**
     * Overrides the printStatus() method of the superclass to print the status of the lamp to a file.
     * Formats the date and time using the DateTimeFormatter class.
     */
    @Override
    public void printStatus() {
        // Creates a DateTimeFormatter object to format the date and time.
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss");
        String formattedDateTime;
        try {
            // Formats the switch time of the lamp using the DateTimeFormatter.
            formattedDateTime = this.getSwitchTime().format(formatter);
        } catch (NullPointerException e) {
            // If the switch time is null, sets the formattedDateTime to "null".
            formattedDateTime = "null";
        }
        String tmpS = String.valueOf(kelvin) + "K";
        
        if (this.colorMode) {
            // If the lamp is in color mode, formats the color code as a hexadecimal string.
            String hex = String.format("0x%06X", colorCode);
            tmpS = hex;
        }
        
        // Writes the formatted string to a file using the FileOutput class.
        FileOutput.writeToFile(Main.output,
                String.format("Smart Color Lamp " + name + " is " +
                (status ? "on" : "off") + " and its color value is %s with "
                        + "%d%% brightness, and its time to switch its status is %s." 
                , tmpS, brightness, formattedDateTime), true, true);
    }

}