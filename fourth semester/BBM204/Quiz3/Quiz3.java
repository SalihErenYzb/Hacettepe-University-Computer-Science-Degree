import java.util.*;
import java.io.*;
public class Quiz3 {
    static int parent[] = new int[1000];
    static int size[] = new int[1000];
    public static int find(int x){
        if(parent[x] == x)return x;
        return parent[x] = find(parent[x]);
    }
    public static int union(int x, int y){
        x = find(x);
        y = find(y);
        if(x == y)return 0;
        if(size[x] < size[y]){
            parent[x] = y;
            size[y] += size[x];
        }else{
            parent[y] = x;
            size[x] += size[y];
        }
        return 1;
    }
    public static void main(String[] args) throws IOException {
        Scanner sc = new Scanner(new File(args[0]));
        for (int t = sc.nextInt(); t > 0; t--) {
            int s = sc.nextInt();
            int p = sc.nextInt();
            ArrayList<int[]> coords = new ArrayList<>();
            ArrayList<Object[]> edges = new ArrayList<>();
            for(int i=0; i<p; i++){
                coords.add(new int[]{sc.nextInt(), sc.nextInt()});
                parent[i] = i;
                size[i] = 1;
                for (int j = 0; j < i; j++) 
                    edges.add(new Object[]{j, i, Math.hypot(coords.get(i)[0] - coords.get(j)[0],
                         coords.get(i)[1] - coords.get(j)[1])});
            }
            Collections.sort(edges, (a, b) -> Double.compare((double)a[2], (double)b[2]));
            for(Object[] edge : edges)
                if (union((int)edge[0], (int)edge[1]) != 0 && --p == s) 
                    System.out.printf("%.2f\n", (double)edge[2]);
        }
        sc.close();
    }
}