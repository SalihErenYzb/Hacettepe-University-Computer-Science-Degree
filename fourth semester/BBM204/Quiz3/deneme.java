import java.util.*;
import java.io.*;

public class deneme {
    static int parent[];
    public static int find(int x){
        if(parent[x] == x)return x;
        return parent[x] = find(parent[x]);
    }
    public static int union(int x, int y){
        if(find(x) != find(y)) 
            return parent[parent[x]] = y;
        return 0;
    }
    public static void solve(Scanner sc) {
        int s = sc.nextInt();
        int p = sc.nextInt();
        ArrayList<int[]> coords = new ArrayList<>();
        parent = new int[p];
        ArrayList<Object[]> edges = new ArrayList<>();
        for(int i=0; i<p; i++){
            coords.add(new int[]{sc.nextInt(), sc.nextInt()});
            parent[i] = i;
            for (int j = 0; j < i; j++) {
                double dist = Math.hypot(coords.get(i)[0] - coords.get(j)[0], coords.get(i)[1] - coords.get(j)[1]);
                edges.add(new Object[]{j, i, dist});
            }
        }
        Collections.sort(edges, (a, b) -> Double.compare((double)a[2], (double)b[2]));
        for(Object[] edge : edges){
            if (union((int)edge[0], (int)edge[1]) != 0 && --p == s) {
                System.out.printf("%.2f\n", (double)edge[2]);
            }
        }
    }
    public static void main(String[] args) throws IOException {
        Scanner sc = new Scanner(new File(args[0]));
        int t = sc.nextInt();
        while(t-- >0){
            solve(sc);
        }
    }
}
