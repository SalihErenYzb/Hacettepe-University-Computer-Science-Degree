import java.util.*;
import java.io.*;

public class Quiz2 {
    public static void main(String[] args) throws IOException {
        Scanner sc = new Scanner(new File(args[0]));

        int M,n;
        M = sc.nextInt();
        n = sc.nextInt();
        int[] weights = new int[n];
        for(int i=0;i<n;i++){
            weights[i] = sc.nextInt();
        }
        sc.close();
        boolean[][] dp = new boolean[M+1][n+1];
        for(int m=0; m <= M; m++){
            for (int i = 0; i <= n; i++) {
                if (i == 0) {
                    if (m == 0) {dp[m][i] = true;}
                    else {dp[m][i] = false;}
                } else {
                    dp[m][i] = dp[m][i-1];
                    if (weights[i - 1] <= m) {
                        dp[m][i] = dp[m][i] || dp[m - weights[i - 1]][i -1];
                    }
                }
            }
        }
        int m = M;
        while (m >= 0 && !dp[m][n]) {
            m--;
        }
        System.out.println(m);
        // print the whole array
        for (int i = 0; i <= M; i++) {
            for (int j = 0; j <= n; j++) {
                if (dp[i][j]) {
                    System.out.print("1");
                } else {
                    System.out.print("0");
                }
            }
            System.out.println();
        }
    }
}
