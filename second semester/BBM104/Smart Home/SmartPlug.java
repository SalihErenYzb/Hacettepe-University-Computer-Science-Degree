import java.time.LocalDateTime;
import java.time.Duration;

class SmartPlug extends SmartDevice {
    private double ampere;
    private double energyConsumed;
    boolean plugged = false;
    LocalDateTime energyStart;
    boolean energyOpen = false;
    
    // Constructor for SmartPlug class that takes a name parameter.
    public SmartPlug(String name){
        super(name);
        this.energyConsumed = 0.0;
    }
    
    // Constructor for SmartPlug class that takes a name and status parameter.
    public SmartPlug(String name,String str ) {
        super(name);
        this.setStatus(str);
    }
    
    // Constructor for SmartPlug class that takes a name, status, and ampere parameter.
    public SmartPlug(String name,String str,String ampere ) {
        super(name);
        this.setStatus(str);
        this.setAmpere(ampere);
        if(this.getStatus()) {
        	// If the plug is on, set energyOpen to true and energyStart to the current time.
        	energyOpen = true;
        	energyStart = Main.time;
        }
    }

    // Getter method for ampere property.
    public double getAmpere() {
        return ampere;
    }

    // Getter method for energyConsumed property.
    public double getEnergyConsumed() {
        return energyConsumed;
    }
    
    // Setter method for status property.
    public void setStatus(boolean status) {
    	if (status) {
            if (this.plugged) {
            	// If the plug is plugged in and turned on, set energyOpen to true and energyStart to the current time.
            	energyOpen=true;
            	energyStart = Main.time;
            }		
    	}else {
            if (this.plugged) {
            	// If the plug is plugged in and turned off, set energyOpen to false and update energyConsumed based on the duration of time it was on.
            	energyOpen=false;
            	updateEnergyConsumed(Duration.between(energyStart, Main.time).toMinutes());
            	}
    	}
        this.status = status;
    }
    
    // Setter method for status property that takes a string parameter.
    public void setStatus(String str) {
        if (str.equalsIgnoreCase("On") && !this.getStatus()) {
            this.setStatus(true);
        } else if (str.equalsIgnoreCase("Off")) {
            this.setStatus(false);
        } else if (str.equalsIgnoreCase("Off") || str.equalsIgnoreCase("On")) {
        	// If the status parameter is not "On" or "Off", throw an IllegalArgumentException.
        	throw new IllegalArgumentException("ERROR: Status is already given value.");
        } else {
            // If the status parameter is not "On" or "Off", throw an IllegalArgumentException.
            throw new IllegalArgumentException("ERROR: Status must be 'On' or 'Off'.");
        }
    }
    
    // Setter method for ampere property that takes a string parameter.
    public void setAmpere(String ampere)throws IllegalArgumentException {
     if (this.plugged == false) {
      try {
    	// Try to parse the ampere parameter as a Double.
    	Double ampere1 = Double.parseDouble(ampere);
        if (ampere1 <= 0) {
            // If the ampere value is not positive, throw an IllegalArgumentException.
            throw new IllegalArgumentException("ERROR: Ampere value must be a positive number!");
        }
        // If the plug is not already plugged in, set plugged to true and set energyOpen and energyStart if the plug is turned on.
        plugged = true;
        if (this.getStatus()) {
        	energyOpen=true;
        	energyStart = Main.time;
        }
        // Set the ampere property to the parsed ampere value.
        this.ampere = ampere1;
      } catch(NumberFormatException  e) {
  		// If the ampere parameter cannot be parsed as a Double, throw an IllegalArgumentException.
  		throw new IllegalArgumentException("ERROR: Given Ampere Value is Not a Integer");
  	}
   } else {
 		// If the plug is already plugged in, throw an IllegalArgumentException.
 		throw new IllegalArgumentException("ERROR: There is already an item plugged in to that plug!");
   }
    }
    
    // Method to unplug a device from the plug.
    public void unplug()throws IllegalArgumentException {
        if (this.plugged == true) {
           plugged = false;
           if (this.getStatus()) {
           	// If the plug was turned on, set energyOpen to false and update energyConsumed based on the duration of time it was on.
        	updateEnergyConsumed(Duration.between(energyStart, Main.time).toMinutes());
           }
      } else {
    		// If the plug is not already plugged in, throw an IllegalArgumentException.
    		throw new IllegalArgumentException("ERROR: This plug has no item to plug out from that plug!");
      }
    }

    // Method to update the energyConsumed property based on the duration of time the plug was on and the ampere rating of the device.
    public void updateEnergyConsumed( double timeInMinutes) {
        energyConsumed += 220 * ampere * timeInMinutes / 60.0;
    }

    // Method to print the status of the SmartPlug.
    @Override
    public void printStatus() {
		String open = "off";
		if (this.getStatus()) {open="on";}
    	FileOutput.writeToFile(Main.output, String.format("Smart Plug %s "
    			+ "is %s and consumed %.2fW so far (excluding current device)",name,
    			open,energyConsumed),true,false);

    	super.printStatus();
    }
}