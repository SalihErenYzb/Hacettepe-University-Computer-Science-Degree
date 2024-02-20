public class SmartLamp extends SmartDevice {

    // Instance variables
    protected int kelvin = 4000;
    protected int brightness = 100;

    // Constructors

    /**
     * Constructs a new SmartLamp object with the given name.
     *
     * @param name The name of the SmartLamp.
     */
    public SmartLamp(String name) {
        super(name);
    }

    /**
     * Constructs a new SmartLamp object with the given name and status.
     *
     * @param name The name of the SmartLamp.
     * @param str  The status of the SmartLamp as a string ("On" or "Off").
     */
    public SmartLamp(String name, String str) {
        super(name);
        this.setStatus(str);
    }

    /**
     * Constructs a new SmartLamp object with the given name, status, kelvin value, and brightness.
     *
     * @param name       The name of the SmartLamp.
     * @param str        The status of the SmartLamp as a string ("On" or "Off").
     * @param kelvin     The kelvin value of the SmartLamp as a string.
     * @param brightness The brightness of the SmartLamp as a string.
     */
    public SmartLamp(String name, String str, String kelvin, String brightness) {
        super(name);
        this.setStatus(str);
        this.setBrightness(brightness);
        this.setKelvin(kelvin);
    }

    // Methods

    /**
     * Sets the status of the SmartLamp based on the given boolean value.
     *
     * @param status The new status of the SmartLamp.
     */
    public void setStatus(boolean status) {
        this.status = status;
    }

    /**
     * Returns the kelvin value of the SmartLamp.
     *
     * @return The kelvin value of the SmartLamp.
     */
    public int getKelvin() {
        return kelvin;
    }

    /**
     * Returns the brightness of the SmartLamp.
     *
     * @return The brightness of the SmartLamp.
     */
    public int getBrightness() {
        return brightness;
    }

    /**
     * Sets the kelvin value of the SmartLamp based on the given string.
     *
     * @param kelvin The new kelvin value of the SmartLamp as a string.
     * @throws IllegalArgumentException if the given kelvin value is not within the range of 2000K-6500K or is not an integer.
     */
    public void setKelvin(String kelvin) throws IllegalArgumentException {
        try {
            // Try to parse the kelvin string into an integer
            int kelvin1 = Integer.parseInt(kelvin);
            // Check if the resulting integer is within the valid range
            if (kelvin1 >= 2000 && kelvin1 <= 6500) {
                // If yes, set the instance variable to the new value
                this.kelvin = kelvin1;
            } else {
                // If no, throw an IllegalArgumentException with an appropriate error message
                throw new IllegalArgumentException("ERROR: Kelvin value must be in range of 2000K-6500K!");
            }
        } catch (NumberFormatException e) {
            // If the parsing fails, throw an IllegalArgumentException with an appropriate error message
            throw new IllegalArgumentException("ERROR: Given Kelvin value is not an integer.");
        }
    }

    /**
     * Sets the brightness of the SmartLamp based on the given string.
     *
     * @param brightness1 The new brightness of the SmartLamp as a string.
     * @throws IllegalArgumentException if the given brightness value is not within the range of 0%-100% or is not an integer.
     */
    public void setBrightness(String brightness1) throws IllegalArgumentException {
        try {
            // Try to parse the brightness string into an integer
            int brightness = Integer.parseInt(brightness1);
            // Check if the resulting integer is within the valid range
            if (brightness >= 0 && brightness <= 100) {
                // If yes, set the instance variable to the new value
                this.brightness = brightness;
            } else {
                // If no, throw an IllegalArgumentException with an appropriate error message
                throw new IllegalArgumentException("ERROR: Brightness must be in range of 0%-100%!");
            }
        } catch (NumberFormatException e) {
            // If the parsing fails, throw an IllegalArgumentException with an appropriate error message
            throw new IllegalArgumentException("ERROR: Given Brightness value is not an integer.");
        }
    }

    /**
     * Prints the status of the SmartLamp to the console and writes it to the output file.
     */
    @Override
    public void printStatus() {
        // Check the current status of the SmartLamp and set the string variable accordingly
        String open = "off";
        if (this.getStatus()) {
            open = "on";
        }

        // Write the status information to the output file in a formatted string
        FileOutput.writeToFile(Main.output,
                String.format("Smart Lamp %s is %s and its kelvin value is "
                                + "%dK with %d%% brightness"
                        , name, open, kelvin, brightness), true, false);
        
        // Call the printStatus() method of the superclass to print the status information to the console
        super.printStatus();
    }
}