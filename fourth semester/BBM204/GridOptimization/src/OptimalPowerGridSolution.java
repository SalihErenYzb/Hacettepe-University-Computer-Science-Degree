import java.util.ArrayList;
import java.util.stream.Collectors;

/**
 * A class that represents the optimal solution for the Power Grid optimization scenario (Dynamic Programming)
 */

public class OptimalPowerGridSolution {
    private int maxNumberOfSatisfiedDemands;
    private ArrayList<Integer> hoursToDischargeBatteriesForMaxEfficiency;
    

    public OptimalPowerGridSolution(int maxNumberOfSatisfiedDemands, ArrayList<Integer> hoursToDischargeBatteriesForMaxEfficiency) {
        this.maxNumberOfSatisfiedDemands = maxNumberOfSatisfiedDemands;
        this.hoursToDischargeBatteriesForMaxEfficiency = hoursToDischargeBatteriesForMaxEfficiency;
        System.out.println("Maximum number of satisfied gigawatts: " + maxNumberOfSatisfiedDemands);
        System.out.print("Hours at which the battery bank should be discharged: ");
        System.out.println(hoursToDischargeBatteriesForMaxEfficiency.stream().map(Object::toString).collect(Collectors.joining(", ")));

    }

    public OptimalPowerGridSolution() {

    }

    public int getmaxNumberOfSatisfiedDemands() {
        return maxNumberOfSatisfiedDemands;
    }

    public ArrayList<Integer> getHoursToDischargeBatteriesForMaxEfficiency() {
        return hoursToDischargeBatteriesForMaxEfficiency;
    }

}
