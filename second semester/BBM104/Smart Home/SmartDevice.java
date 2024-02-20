import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Represents a smart device that can be turned on or off and scheduled to switch at a specific time.
 */
abstract class SmartDevice {
    protected String name; // Name of the smart device
    protected boolean status = false;// Status of the smart device (on/off)
    protected LocalDateTime switchTime = null; // Time at which the smart device is scheduled to switch
    /**
     * Constructs a new SmartDevice with the given name.
     *
     * @param name The name of the SmartDevice.
     */
    public SmartDevice(String name) {
        this.name = name;
    }
    /**
     * Returns the name of the SmartDevice.
     *
     */
    public String getName() {
        return name;
    }
    /**
     * Sets the name of the SmartDevice.
     *
     * @param name The new name of the SmartDevice.
     */
    public void setName(String name) {
        this.name= name;
    }
    /**
     *
     * @return The time at which the SmartDevice is scheduled to switch.
     */
    public LocalDateTime getSwitchTime() {
        return switchTime;
    }


    /**
     * Removes the scheduled switch time for the SmartDevice and updates its status accordingly.
     * Throws an IllegalArgumentException if the SmartDevice has no switch time.
     */
    public void removeSwitchTime() {

        // Check if the SmartDevice has a scheduled switch time
        if (!this.hasSwitchTime()) {
            throw new IllegalArgumentException("ERROR: Device Has No Switch Time");
        }

        // Remove the scheduled switch time
        this.switchTime = null;

        // Update the status of the SmartDevice
        if (this.getStatus()==false) {
            this.setStatus(true);
        } else {
            this.setStatus(false);
        }

        // Find the index of the first occurrence of a device with a null switch time in the list of all devices
        int index = -1;
        for (int i = 0; i < Main.devices.size(); i++) {
            SmartDevice d = Main.devices.get(i);
            if (d != null && !d.hasSwitchTime()) {
                index = i;
                break;
            }
        }

        // If no device with a null switch time was found, add the device to the end of the list
        if (index == -1) {
            Main.devices.add(this);
        } else {
            // Otherwise, add the device before the first occurrence of a device with a null switch time
            Main.devices.add(index, this);
        }
    }
    /**
     * Returns true if the SmartDevice has a scheduled switch time.
     *
     * @return True if the SmartDevice has a scheduled switch time.
     */
    public boolean hasSwitchTime() {
        return switchTime != null;
    }
    /**
     * Prints the status of the SmartDevice to a file.
     * If the SmartDevice has a scheduled switch time, also prints the time.
     *
     * @throws DateTimeParseException if the scheduled switch time cannot be formatted as a string.
     */
    public void printStatus() {

        // Format the scheduled switch time, if it exists
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss");
        String formattedDateTime;
        try {
            formattedDateTime = this.getSwitchTime().format(formatter);
        } catch (NullPointerException e) {
            // If the scheduled switch time is null, set the formatted time to "null"
            formattedDateTime = "null";
        }

        // Write the status and scheduled switch timeX (if it exists) to a file
        FileOutput.writeToFile(Main.output,
                String.format(", and its time to switch its status is %s.", formattedDateTime), true, true);
    }
    /**
     * Sets the status of the SmartDevice to the given value.
     *
     * @param status The new status of the SmartDevice.
     */
    public void setStatus(boolean status) {
        this.status = status;
    }

    /**
     * Sets the status of the SmartDevice based on the given input string.
     * Throws an IllegalArgumentException if the input string is not "On" or "Off", or if the SmartDevice's status is already the same as the input.
     *
     * @param str The input string representing the desired status ("On" or "Off").
     */
    public void setStatus(String str) {

        // Check if the input string is valid and update the status of the SmartDevice accordingly
        if (str.equalsIgnoreCase("On") && !this.getStatus()) {
            this.setStatus(true);
        } else if (str.equalsIgnoreCase("Off") && this.getStatus()) {
            this.setStatus(false);
        } else if (str.equalsIgnoreCase("On") || str.equalsIgnoreCase("Off")) {
            // If the desired status is already the same as the current status, throw an exception
            throw new IllegalArgumentException("ERROR: Status is already same");
        } else {
            // If the input string is not "On" or "Off", throw an exception
            throw new IllegalArgumentException("ERROR: Erroneous command!");
        }
    }
    /**
     * Returns the status of the SmartDevice as a boolean value based on the given input string.
     *
     * @param str The input string representing the desired status ("On" or "Off").
     * @return The status of the SmartDevice as a boolean value.
     * @throws IllegalArgumentException if the input string is not "On" or "Off".
     */
    public boolean getStatus(String str) {

        // Check if the input string is valid and return the corresponding boolean value
        if (str.equalsIgnoreCase("On")) {
            return true;
        } else if (str.equalsIgnoreCase("Off")) {
            return false;
        } else {
            // If the input string is not "On" or "Off", throw an exception
            throw new IllegalArgumentException("ERROR: Erroneous command!");
        }
    }
    /**
     * Returns the status of the SmartDevice as a boolean value.
     *
     * @return The status of the SmartDevice as a boolean value.
     */
    public boolean getStatus() {
        return status;
    }
    /**
     * Sets the scheduled switch time for the SmartDevice to the given LocalDateTime object.
     * If the given time is in the past, throws an IllegalArgumentException.
     * If the SmartDevice already has a scheduled switch time, removes the old switch time and inserts the SmartDevice in the correct position in the list of devices with switch times.
     * If the SmartDevice does not have a scheduled switch time, adds the SmartDevice to the end of the list of devices with switch times.
     *
     * @param switchTime The new scheduled switch time for the SmartDevice.
     * @throws IllegalArgumentException if the given time is in the past.
     */
    public void setSwitchTime(LocalDateTime switchTime) {

        // Check if the given time is in the past and throw an exception if it is
        if (switchTime.isBefore(Main.time)) {
            throw new IllegalArgumentException("ERROR: Switch time cannot be in the past!");
        }

        // Set the new switch time and determine if the SmartDevice already had a switch time
        boolean hadSwitchTime = (this.switchTime != null);
        boolean leftOrRight;
        if (hadSwitchTime) {
            leftOrRight = switchTime.isAfter(this.switchTime);
        } else {
            leftOrRight = false;
        }
        this.switchTime = switchTime;
        SmartDevice deviceTmp = this;
        int index = 0;

        // If the new switch time is equal to the current time, remove the switch time for the SmartDevice
        if (switchTime.equals(Main.time)) {
            this.removeSwitchTime();
        } else {
            // Find the correct position to insert the SmartDevice in the list of devices with switch times
            while (index < Main.devices.size()) {
                if (leftOrRight && Main.devices.get(index).getSwitchTime().equals(switchTime)) {
                    // If there is already a device with the same switch time, insert the new device after it
                    break;
                }
                if (Main.devices.get(index).getSwitchTime() == null || Main.devices.get(index).getSwitchTime().isAfter(switchTime)) {
                    // If the device at the current index has a switch time later than the new switch time, insert the new device before it
                    break;
                }
                index++;
            }

            // Insert the new device at the correct position
            Main.devices.add(index, deviceTmp);
        }

        // If the SmartDevice already had a switch time, remove it from the list of devices with switch times
        if (hadSwitchTime) {
            this.removeSwitchTime();
        }
    }
    /**
     * Sets the scheduled switch time for the SmartDevice to the LocalDateTime object parsed from the given input string.
     * Throws a DateTimeParseException if the input string cannot be parsed as a LocalDateTime object.
     *
     * @param switchTimeString The input string representing the new scheduled switch time for the SmartDevice.
     * @throws DateTimeParseException if the input string cannot be parsed as a LocalDateTime object.
     */
    public void setSwitchTime(String switchTimeString) {
        LocalDateTime switchTime = LocalDateTime.parse(switchTimeString, DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss"));
        setSwitchTime(switchTime);
    }
}
