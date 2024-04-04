import java.util.ArrayList;
import java.util.Collections;

/**
 * This class accomplishes Mission Eco-Maintenance
 */
public class OptimalESVDeploymentGP
{

    /*
     * Should include tasks assigned to ESVs.
     * For the sample input:
     * 8 100
     * 20 50 40 70 10 30 80 100 10
     * 
     * The list should look like this:
     * [[100], [80, 20], [70, 30], [50, 40, 10], [10]]
     * 
     * It is expected to be filled after getMinNumESVsToDeploy() is called.
     */
    private ArrayList<ArrayList<Integer>> maintenanceTasksAssignedToESVs = new ArrayList<>();
    private ArrayList<Integer> maintenanceTaskEnergyDemands;

    ArrayList<ArrayList<Integer>> getMaintenanceTasksAssignedToESVs() {
        return maintenanceTasksAssignedToESVs;
    }

    public OptimalESVDeploymentGP(ArrayList<Integer> maintenanceTaskEnergyDemands) {
        this.maintenanceTaskEnergyDemands = maintenanceTaskEnergyDemands;
    }

    public ArrayList<Integer> getMaintenanceTaskEnergyDemands() {
        return maintenanceTaskEnergyDemands;
    }

    /**
     *
     * @param maxNumberOfAvailableESVs the maximum number of available ESVs to be deployed
     * @param maxESVCapacity the maximum capacity of ESVs
     * @return the minimum number of ESVs required using first fit approach over reversely sorted items.
     * Must return -1 if all tasks can't be satisfied by the available ESVs
     */ 
    public int getMinNumESVsToDeploy(int maxNumberOfAvailableESVs, int maxESVCapacity)
    {
        //sort the tasks in descending order
        Collections.sort(maintenanceTaskEnergyDemands, Collections.reverseOrder());
        int numberOfESVs = 0;
        ArrayList<Integer> Sums = new ArrayList<>();
        for (int j = 0; j < maintenanceTaskEnergyDemands.size(); j++) {
            int foundIndex = -1;
            for (int i = 0; i < maintenanceTasksAssignedToESVs.size(); i++) {
                if (Sums.get(i)+ maintenanceTaskEnergyDemands.get(j) <= maxESVCapacity) {
                    foundIndex = i;
                    break;
                }
            }
            if (foundIndex == -1) {
                if (numberOfESVs == maxNumberOfAvailableESVs) {
                    System.out.println("Warning: Mission Eco-Maintenance Failed.");
                    return -1;
                }         
                if (maintenanceTaskEnergyDemands.get(j) > maxESVCapacity) {
                    System.out.println("Warning: Mission Eco-Maintenance Failed.");
                    return -1;
                }             
                ArrayList<Integer> newESV = new ArrayList<>();
                newESV.add(maintenanceTaskEnergyDemands.get(j));
                maintenanceTasksAssignedToESVs.add(newESV);
                Sums.add(maintenanceTaskEnergyDemands.get(j));
                numberOfESVs++;
            } else {
                maintenanceTasksAssignedToESVs.get(foundIndex).add(maintenanceTaskEnergyDemands.get(j));
                Sums.set(foundIndex, Sums.get(foundIndex) + maintenanceTaskEnergyDemands.get(j));
            }   
        }

        System.out.println("The minimum number of ESVs to deploy: "+numberOfESVs);
        for (int i = 0; i < maintenanceTasksAssignedToESVs.size(); i++) {
            System.out.println("ESV " + (i + 1) + " tasks: " + maintenanceTasksAssignedToESVs.get(i));
        }
        return numberOfESVs;
    

    }

}
