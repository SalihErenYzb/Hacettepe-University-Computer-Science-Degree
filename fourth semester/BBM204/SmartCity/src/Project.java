import java.io.Serializable;
import java.util.*;

public class Project implements Serializable {
    static final long serialVersionUID = 33L;
    private final String name;
    private final List<Task> tasks;
    private HashMap<Integer, Integer> taskDuration = new HashMap<>();

    public Project(String name, List<Task> tasks) {
        this.name = name;
        this.tasks = tasks;
    }

    private int fillTaskDuration(int i) {
        // taskDuration[i] = max(taskDuration[dependencies]) + duration
        // basecase taskDuration[0] = duration[0]
        if (i == 0) {
            taskDuration.put(i, tasks.get(i).getDuration());
            return tasks.get(i).getDuration();
        }else if (taskDuration.containsKey(i)){
            return taskDuration.get(i);
        }
        else{
            Task task = tasks.get(i);
            int duration = task.getDuration();
            List<Integer> dependencies = task.getDependencies();
            int max = 0;
            for (int d : dependencies) {
                // if taskDuration exists use it else calculate it
                fillTaskDuration(d);
                max = Math.max(max, taskDuration.get(d));
            }
            taskDuration.put(i, max + duration);
            return max + duration;
        }
    }
    /**
     * @return the total duration of the project in days
     */
    public int getProjectDuration() {
        int[] schedule = getEarliestSchedule();
        return schedule[schedule.length - 1] + tasks.get(schedule.length - 1).getDuration();
    }

    /**
     * Schedule all tasks within this project such that they will be completed as early as possible.
     *
     * @return An integer array consisting of the earliest start days for each task.
     */
    public int[] getEarliestSchedule() {

        ArrayList<Task> tasks = new ArrayList<>(this.tasks);
        int last = tasks.size() - 1;
        for (int k = last; k >= 0; k--){
            fillTaskDuration(k);
        }
        int[] schedule = new int[tasks.size()];
        for (int i = 0; i < tasks.size(); i++) {
            schedule[i] = taskDuration.get(i) - tasks.get(i).getDuration();
        }
        return schedule;
    }

    public static void printlnDash(int limit, char symbol) {
        for (int i = 0; i < limit; i++) System.out.print(symbol);
        System.out.println();
    }
    public void printProject() {
        System.out.println("Project name: " + name);
        System.out.println("Tasks:");
        for (Task t : tasks) {
            System.out.println("taskID: " + t.getTaskID()); 
            System.out.println("description: " + t.getDescription());
            System.out.println("duration: " + t.getDuration());
            System.out.println("dependencies: " + t.getDependencies());
            System.out.println();
        }
    }   

    /**
     * Some free code here. YAAAY! 
     */
    public void printSchedule(int[] schedule) {
        int limit = 65;
        char symbol = '-';
        printlnDash(limit, symbol);
        System.out.println(String.format("Project name: %s", name));
        printlnDash(limit, symbol);

        // Print header
        System.out.println(String.format("%-10s%-45s%-7s%-5s","Task ID","Description","Start","End"));
        printlnDash(limit, symbol);
        for (int i = 0; i < schedule.length; i++) {
            Task t = tasks.get(i);
            System.out.println(String.format("%-10d%-45s%-7d%-5d", i, t.getDescription(), schedule[i], schedule[i]+t.getDuration()));
        }
        printlnDash(limit, symbol);
        System.out.println(String.format("Project will be completed in %d days.", tasks.get(schedule.length-1).getDuration() + schedule[schedule.length-1]));
        printlnDash(limit, symbol);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Project project = (Project) o;

        int equal = 0;

        for (Task otherTask : ((Project) o).tasks) {
            if (tasks.stream().anyMatch(t -> t.equals(otherTask))) {
                equal++;
            }
        }

        return name.equals(project.name) && equal == tasks.size();
    }

}
