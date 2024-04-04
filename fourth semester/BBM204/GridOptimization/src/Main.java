import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Main class
 */
// FREE CODE HERE
public class Main {
    public static void main(String[] args) throws IOException {

       /** MISSION POWER GRID OPTIMIZATION BELOW **/

        System.out.println("##MISSION POWER GRID OPTIMIZATION##");
        // You are expected to read the file given as the first command-line argument to read 
        // the energy demands arriving per hour. Then, use this data to instantiate a 
        // PowerGridOptimization object. You need to call getOptimalPowerGridSolutionDP() method
        // of your PowerGridOptimization object to get the solution, and finally print it to STDOUT.
        //Scanner scanner = new Scanner(new File("demandSchedule.dat"));
        Scanner scanner = new Scanner(new File(args[0]));

        ArrayList<Integer> amountOfEnergyDemandsArrivingPerHour = new ArrayList<>();
        while (scanner.hasNextInt()) {
            amountOfEnergyDemandsArrivingPerHour.add(scanner.nextInt());
        }
        int total = 0;
        for (int i = 0; i < amountOfEnergyDemandsArrivingPerHour.size(); i++){
            total += amountOfEnergyDemandsArrivingPerHour.get(i);
        }
        PowerGridOptimization powerGridOptimization = new PowerGridOptimization(amountOfEnergyDemandsArrivingPerHour);
        OptimalPowerGridSolution optimal = powerGridOptimization.getOptimalPowerGridSolutionDP();
        System.out.println("The number of unsatisfied gigawatts: " +(total-optimal.getmaxNumberOfSatisfiedDemands()) );
        scanner.close();
        System.out.println("##MISSION POWER GRID OPTIMIZATION COMPLETED##");

        /** MISSION ECO-MAINTENANCE BELOW **/

        System.out.println("##MISSION ECO-MAINTENANCE##");
        // You are expected to read the file given as the second command-line argument to read
        // the number of available ESVs, the capacity of each available ESV, and the energy requirements 
        // of the maintenance tasks. Then, use this data to instantiate an OptimalESVDeploymentGP object.
        // You need to call getMinNumESVsToDeploy(int maxNumberOfAvailableESVs, int maxESVCapacity) method
        // of your OptimalESVDeploymentGP object to get the solution, and finally print it to STDOUT.
        Scanner scanner2 = new Scanner(new File(args[1]));
        int maxNumberOfAvailableESVs = scanner2.nextInt();
        int maxESVCapacity = scanner2.nextInt();
        ArrayList<Integer> maintenanceTaskEnergyDemands = new ArrayList<>();
        while (scanner2.hasNextInt()) {
            maintenanceTaskEnergyDemands.add(scanner2.nextInt());
        }
        OptimalESVDeploymentGP optimalESVDeploymentGP = new OptimalESVDeploymentGP(maintenanceTaskEnergyDemands);
        optimalESVDeploymentGP.getMinNumESVsToDeploy(maxNumberOfAvailableESVs, maxESVCapacity);
        scanner2.close();
        System.out.println("##MISSION ECO-MAINTENANCE COMPLETED##");
    }
}
