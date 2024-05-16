import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;


import java.util.HashMap;

class UrbanTransportationApp implements Serializable {
    static final long serialVersionUID = 99L;
    
    public HyperloopTrainNetwork readHyperloopTrainNetwork(String filename) {
        HyperloopTrainNetwork hyperloopTrainNetwork = new HyperloopTrainNetwork();
        hyperloopTrainNetwork.readInput(filename);
        return hyperloopTrainNetwork;
    }

    /**
     * Function calculate the fastest route from the user's desired starting point to 
     * the desired destination point, taking into consideration the hyperloop train
     * network. 
     * @return List of RouteDirection instances
     */
    public List<RouteDirection> getFastestRouteDirections(HyperloopTrainNetwork network) {
        List<RouteDirection> routeDirections = new ArrayList<>();
        
        // do djikstra's algorithm here 
        // and save the route using hashmap
        HashMap<Station, Object[]> routes = new HashMap<>(); //objec[] has (station, isTrain)

        // make priority queue of (cost, station, oldstation) cost is a Double   
        PriorityQueue<Object[]> pq = new PriorityQueue<>((a,b) -> Double.compare((Double)a[0], (Double)b[0]));
        pq.add(new Object[]{0.0, network.startPoint, network.startPoint,false});



        while (!pq.isEmpty()) {
            Object[] current = pq.poll();

            Station currentStation = (Station) current[1];
            double currentCost = (Double) current[0];
            Station oldStation = (Station) current[2];
            if (routes.containsKey(currentStation)) {
                continue;
            }
            routes.put(currentStation, new Object[]{oldStation, (boolean) current[3]});

            if (currentStation.equals(network.destinationPoint)) {
                break;
            }
            // can walk to every station
            // or if already in a train line, use train line to get to next station
            for (TrainLine trainLine: network.lines){
                // if trainline contains current station find the next station in the trainline
                if (trainLine.trainLineStations.contains(currentStation)){
                    int index = trainLine.trainLineStations.indexOf(currentStation);
                    if (index != trainLine.trainLineStations.size() - 1){
                        Station nextStation = trainLine.trainLineStations.get(index + 1);
                        double cost = currentCost + findDist(currentStation, nextStation) / network.averageTrainSpeed;
                        pq.add(new Object[]{cost, nextStation, currentStation,true});
                    }
                    if (index != 0){
                        Station prevStation = trainLine.trainLineStations.get(index - 1);
                        double cost = currentCost + findDist(currentStation, prevStation) / network.averageTrainSpeed;
                        pq.add(new Object[]{cost, prevStation, currentStation,true});
                    }
                }
                // walk to every station
                for (Station station: trainLine.trainLineStations){
                    double cost = currentCost + findDist(currentStation, station) / network.averageWalkingSpeed;
                    pq.add(new Object[]{cost, station, currentStation,false});
                }
                
            }
            // add the destination point to the priority queue as walkable
            double cost = currentCost + findDist(currentStation, network.destinationPoint) / network.averageWalkingSpeed;
            pq.add(new Object[]{cost, network.destinationPoint, currentStation,false});
            // print the added walking to destination point
            // System.out.println("Added walking to destination point");
            // System.out.println("Current Cost: " + currentCost + ", Cost: " + cost);
            
            // System.out.println("Current Station: " + currentStation.description+"\n");
        }
        // now from destination point we can find the route in reverse
        Station current = network.destinationPoint;
        while (!current.equals(network.startPoint)){
            Object[] route = routes.get(current);
            Station oldStation = (Station) route[0];
            boolean isTrain = (boolean) route[1];
            if (isTrain){
                routeDirections.add(new RouteDirection(oldStation.description, current.description, findDist(oldStation, current) / network.averageTrainSpeed, true));
            }else{
                routeDirections.add(new RouteDirection(oldStation.description, current.description, findDist(oldStation, current) / network.averageWalkingSpeed, false));
            }
            current = oldStation;
        }
        // reverse the routeDirections
        List<RouteDirection> reversedRouteDirections = new ArrayList<>();
        for (int i = routeDirections.size() - 1; i >= 0; i--){
            reversedRouteDirections.add(routeDirections.get(i));
        }
        return reversedRouteDirections;
    }
    private double findDist(Station a, Station b){
        return Math.sqrt(Math.pow(a.coordinates.x - b.coordinates.x, 2) + Math.pow(a.coordinates.y - b.coordinates.y, 2));
    }
    /**
     * Function to print the route directions to STDOUT
     */
    public void printRouteDirections(List<RouteDirection> directions) {
        /*The fastest route takes 8 minute(s).
        Directions
        ----------*/
        //find the total time
        double totalTime = 0;
        for (RouteDirection routeDirection: directions){
            totalTime += routeDirection.duration;
        }
        int minutes = (int)totalTime;
        if (totalTime - minutes >= 0.5){
            minutes++;
        }
        System.out.println("The fastest route takes " + minutes + " minute(s).");
        System.out.println("Directions");
        System.out.println("----------");
        // print the route directions
        for (int i = 0; i  < directions.size(); i++){
            RouteDirection routeDirection = directions.get(i);
            System.out.print((i+1) + ".");
            if (routeDirection.trainRide){
                System.out.print(" Get on the train from ");
            }else{
                System.out.print(" Walk from ");
            }
            // write the duration with only 2 decimal points
            double duration = routeDirection.duration;
            duration = Math.round(duration * 100.0) / 100.0;
            String durationString = String.format("%.2f", duration);
            System.out.println("\"" + routeDirection.startStationName+"\"" + " to " + "\""+routeDirection.endStationName+"\"" + " for " + durationString + " minutes.");
        }


    }
}