
/**

The SmartCamera class represents a smart camera device that extends the SmartDevice class.
It has the ability to set and get the number of megabytes per minute, the amount of storage used,
and the current status of the device. It also has the ability to update the storage used based on time,
and print the current status of the device.
*/
import java.time.LocalDateTime;
import java.time.Duration;
class SmartCamera extends SmartDevice {
	private double megabytesPerMinute;
	private double storageUsed;
	private LocalDateTime tempTime;
	/**
	 * Creates a SmartCamera object with the given name and megabytes per minute.
	 * @param name the name of the smart camera
	 * @param mgb the number of megabytes per minute
	 * @throws IllegalArgumentException if the given megabyte value is not a positive number
	 */
	public SmartCamera(String name, String mgb) throws IllegalArgumentException {
	    super(name);
	    this.setMegabytesPerMinute(mgb);
	    this.storageUsed = 0.0;
	}

	/**
	 * Creates a SmartCamera object with the given name, megabytes per minute, and status.
	 * @param name the name of the smart camera
	 * @param mgb the number of megabytes per minute
	 * @param stat the status of the smart camera ("On" or "Off")
	 * @throws IllegalArgumentException if the given megabyte value is not a positive number
	 */
	public SmartCamera(String name, String mgb, String stat) throws IllegalArgumentException {
	    super(name);
	    this.setMegabytesPerMinute(mgb);
	    this.storageUsed = 0.0;
	    if (stat.equals("On")) {
	        this.setStatus(stat);
	    }
	}

	/**
	 * Sets the status of the smart camera and updates the storage used if the device is turned off.
	 * @param status the status of the smart camera
	 */
	public void setStatus(boolean status) {
	    if (status) {
	        tempTime = Main.time;
	    } else {
	        updateStorageUsed(Duration.between(tempTime, Main.time).toMinutes());
	    }
	    this.status = status;
	}

	/**
	 * Gets the number of megabytes per minute.
	 * @return the number of megabytes per minute
	 */
	public double getMegabytesPerMinute() {
	    return megabytesPerMinute;
	}

	/**
	 * Gets the amount of storage used.
	 * @return the amount of storage used
	 */
	public double getStorageUsed() {
	    return storageUsed;
	}

	/**
	 * Sets the number of megabytes per minute.
	 * @param megabytesPerMinute1 the number of megabytes per minute as a string
	 * @throws IllegalArgumentException if the given megabyte value is not a positive number or not an integer
	 */
	public void setMegabytesPerMinute(String megabytesPerMinute1) throws IllegalArgumentException {
	    try {
	        double megabytesPerMinute = Double.parseDouble(megabytesPerMinute1);
	        if (megabytesPerMinute <= 0) {
	            throw new IllegalArgumentException("ERROR: Megabyte value must be a positive number!");
	        }
	        this.megabytesPerMinute = megabytesPerMinute;
	    } catch (NumberFormatException e) {
	        throw new IllegalArgumentException("ERROR: Given Megabyte Value is Not a Integer");
	    }
	}

	/**
	 * Sets the amount of storage used.
	 * @param storageUsed the amount of storage used
	 */
	public void setStorageUsed(double storageUsed) {
	    this.storageUsed = storageUsed;
	}

	/**
	 * Updates the storage used based on the given time in minutes.
	 * @param timeInMinutes the time in minutes
	 */
	public void updateStorageUsed(double timeInMinutes) {
	    storageUsed += megabytesPerMinute * timeInMinutes;
	}

	/**
	 * Prints the current status of the smart camera.
	 */
	@Override
	public void printStatus() {
	    String open = "off";
	    if (this.getStatus()) {
	        open = "on";
	    }
	    FileOutput.writeToFile(Main.output, String.format("Smart Camera %s is %s and used %.2f MB of storage so far (excluding current status)", name, open, storageUsed), true, false);
	    super.printStatus();
	}
}