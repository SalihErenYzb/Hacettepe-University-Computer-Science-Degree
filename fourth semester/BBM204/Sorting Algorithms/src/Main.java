import org.knowm.xchart.*;
import org.knowm.xchart.style.Styler;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;


public class Main {
    static int[] inputAxis = {500, 1000, 2000, 4000, 8000, 16000, 32000 , 64000, 128000, 250000};//
    public static void main(String args[]) throws IOException {

        // X axis data
        
        // read the TrafficFlowDataset.csv file from lib folder
        //String filename = "lib/TrafficFlowDataset.csv";
        String filename = args[0];
        int[] durations = readCSV(filename);
        double[][] yAxis = new double[3][10];        
        // Sorting Experiment
        System.out.println("Sorting Experiment on Random Data");
        yAxis[0] = doExperiment("InsertionSort", durations,false,10);
        yAxis[1] = doExperiment("MergeSort", durations,false,10);
        yAxis[2] = doExperiment("CountingSort", durations,false,10);

        //Save the char as .png and show it
        showAndSaveChart("Random Data", inputAxis, yAxis,true);

        // Sorted data
        System.out.println("Sorting Experiment on Sorted Data");
        Arrays.sort(durations);
        yAxis[0] = doExperiment("InsertionSort", durations,false,10);
        yAxis[1] = doExperiment("MergeSort", durations,false,10);
        yAxis[2] = doExperiment("CountingSort", durations,false,10);
        showAndSaveChart("Sorted Data", inputAxis, yAxis,true);

        // Reverse sorted data
        int[] reverseDurations = new int[durations.length];
        for (int i = 0; i < durations.length; i++) {
            reverseDurations[i] = durations[durations.length - 1 - i];
        }
        System.out.println("Sorting Experiment on Reverse Sorted Data");
        yAxis[0] = doExperiment("InsertionSort", reverseDurations,false,10);
        yAxis[1] = doExperiment("MergeSort", reverseDurations,false,10);
        yAxis[2] = doExperiment("CountingSort", reverseDurations,false,10);
        showAndSaveChart("Reverse Sorted Data", inputAxis, yAxis,true);
        
        
        // Search Experiment
        // linear search
        System.out.println("Search Experiment on Random Data");
        durations = readCSV(filename);
        yAxis[0] = doExperiment("LinearSearch", durations,true,1000);
        //Search on sorted data
        Arrays.sort(durations); 
        System.out.println("Search Experiment on Sorted Data");
        yAxis[1] = doExperiment("LinearSearch", durations,true,1000);
        yAxis[2] = doExperiment("BinarySearch", durations,true,1000); 
        //Save the char as .png and show it
        showAndSaveChart("Search Algorithms", inputAxis, yAxis,false);

    }
    public static double[] doExperiment(String experiment,int[] durations,boolean isNano , int numOfExperiments){
        double[] results = new double[inputAxis.length];
        for (int i = 0; i < inputAxis.length; i++) {
            int input = inputAxis[i];
            int[] arr = Arrays.copyOfRange(durations, 0, input);
            if (experiment.equals("CountingSort")){
                int maxEl = Arrays.stream(arr).max().getAsInt();
                System.out.println("Max value of array is " + maxEl);
            }
            
            for (int j = 0; j < numOfExperiments; j++){
                arr = Arrays.copyOfRange(durations, 0, input);
                long startTime = System.nanoTime();
                selectMethod(arr, experiment);
                long endTime = System.nanoTime();
                results[i] += (double)(endTime - startTime);

            }
            if (!isNano){
                results[i] /= 1000000;
            }
            results[i] /= numOfExperiments;
            results[i] = Math.round(results[i]);
            String unit = isNano ? "ns" : "ms";
            System.out.println(experiment+ ": " + input + " elements: " + (int) results[i] + unit);
        }
        System.out.println("-----------------------------------------");
        return results;
    }
    public static void selectMethod(int[] arr, String experiment){
        Random random = new Random();
        int numToBeSearched = random.nextInt(arr.length);
        numToBeSearched = arr[numToBeSearched];

        if (experiment.equals("InsertionSort")) {
            SearchAndSort.doInsertionSort(arr);
        } else if (experiment.equals("MergeSort")) {
            SearchAndSort.doMergeSort(arr);
        } else if (experiment.equals("CountingSort")) {
            SearchAndSort.doCountingSort(arr);
        }else if (experiment.equals("LinearSearch")) {
            SearchAndSort.doLinearSearch(arr, numToBeSearched);
        }else if (experiment.equals("BinarySearch")) {
            SearchAndSort.doBinarySearch(arr, numToBeSearched);
        }
       
    }
    public static int[] readCSV(String filename) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filename));
        String line = br.readLine();
        ArrayList<Integer> Durations = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            Durations.add(Integer.parseInt(line.split(",")[6]));
        }
        br.close();
        // Convert ArrayList to int[]
        int[] Durations2 = new int[Durations.size()];
        for (int i = 0; i < Durations.size(); i++) {
            Durations2[i] = Durations.get(i);
        }
        return Durations2;
    }
    public static void showAndSaveChart(String title, int[] xAxis, double[][] yAxis,boolean isSort) throws IOException {
        // Create Chart
        String chartTitle = isSort ? "Time in Milliseconds" : "Time in Nanoseconds";
        XYChart chart = new XYChartBuilder().width(800).height(600).title(title)
                .yAxisTitle(chartTitle).xAxisTitle("Input Size").build();

        // Convert x axis to double[]
        double[] doubleX = Arrays.stream(xAxis).asDoubleStream().toArray();        

        // Customize Chart
        chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNE);
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);

        // Add a plot for a sorting algorithm
        if (isSort){
            chart.addSeries("Insertion Sort", doubleX, yAxis[0]);
            chart.addSeries("Merge Sort", doubleX, yAxis[1]);
            chart.addSeries("Counting Sort", doubleX, yAxis[2]);
        }else{
            chart.addSeries("Linear Search on Random Data", doubleX, yAxis[0]);
            chart.addSeries("Linear Search on Sorted Data", doubleX, yAxis[1]);
            chart.addSeries("Binary Search on Sorted Data", doubleX, yAxis[2]);
        }


        // Save the chart as PNG
        BitmapEncoder.saveBitmap(chart, title + ".png", BitmapEncoder.BitmapFormat.PNG);

        // Show the chart
        new SwingWrapper<XYChart>(chart).displayChart();
    }
}
