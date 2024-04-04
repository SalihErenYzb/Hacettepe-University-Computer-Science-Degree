import java.util.ArrayList;

/**
 * This class accomplishes Mission POWER GRID OPTIMIZATION
 */
public class PowerGridOptimization {
    private ArrayList<Integer> amountOfEnergyDemandsArrivingPerHour;

    public PowerGridOptimization(ArrayList<Integer> amountOfEnergyDemandsArrivingPerHour){
        this.amountOfEnergyDemandsArrivingPerHour = amountOfEnergyDemandsArrivingPerHour;
        int sum = 0;
        for (int i = 0; i < amountOfEnergyDemandsArrivingPerHour.size(); i++){
            sum += amountOfEnergyDemandsArrivingPerHour.get(i);
        }
        System.out.println("The total number of demanded gigawatts: " + sum);
    }

    public ArrayList<Integer> getAmountOfEnergyDemandsArrivingPerHour() {
        return amountOfEnergyDemandsArrivingPerHour;
    }
    /**
     *     Function to implement the given dynamic programming algorithm
     *     SOL(0) <- 0
     *     HOURS(0) <- [ ]
     *     For{j <- 1...N}
     *         SOL(j) <- max_{0<=i<j} [ (SOL(i) + min[ D(j), E(j − i) ] ]
     *         HOURS(j) <- [HOURS(i), j]
     *     EndFor
     *
     * @return OptimalPowerGridSolution
     */
    private int getHelper(int j,int i){
        int E = (j-i)*(j-i);
        return Math.min(amountOfEnergyDemandsArrivingPerHour.get(j-1),E);
    }
    public OptimalPowerGridSolution getOptimalPowerGridSolutionDP(){
        // Implement the given dynamic programming algorithm
        int N = amountOfEnergyDemandsArrivingPerHour.size();
        int[] SOL = new int[N+1];
        ArrayList<ArrayList<Integer>> HOURS = new ArrayList<>();
        SOL[0] = 0;
        HOURS.add(new ArrayList<>());
        for(int j = 1; j <= N; j++){
            int maxIndex = 0;
            for (int i = 0; i < j; i++){
                int temp = SOL[i] + getHelper(j,i);
                if (temp > SOL[j]){
                    SOL[j] = temp;
                    maxIndex = i;
                }
            }
            ArrayList<Integer> temp = new ArrayList<>(HOURS.get(maxIndex));
            temp.add(j);
            HOURS.add(temp);
        }

        return new OptimalPowerGridSolution(SOL[N],HOURS.get(N));
    }
}
