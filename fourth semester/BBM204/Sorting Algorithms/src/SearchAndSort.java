
public class SearchAndSort {
    public static void doInsertionSort(int[] arr) {
        for (int i = 1; i < arr.length; i++) {
            int key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }
    public static void doMergeSort(int[] arr) {
        doMergeSort(arr, 0, arr.length - 1);
    }
    private static void doMerge(int[] arr, int l, int m, int r) {
        int n1 = m - l + 1, n2 = r - m;;
        int L[] = new int[n1];
        int R[] = new int[n2];
        for (int i = 0; i < n1; ++i) {
            L[i] = arr[l + i];
        }
        for (int j = 0; j < n2; ++j) {
            R[j] = arr[m + 1 + j];
        }
        int i = 0, j = 0, k = l;
        while (i < n1 && j < n2) {
            if (L[i] <= R[j]) {
                arr[k++] = L[i++];
                continue;
            }
            arr[k++] = R[j++];
        }
        while (i < n1) { arr[k++] = L[i++]; }
        while (j < n2) { arr[k++] = R[j++]; }
    }
    public static void doMergeSort(int[] arr, int l, int r) {
        if (l < r) {
            int m = (l + r) / 2;
            doMergeSort(arr, l, m);
            doMergeSort(arr, m + 1, r);
            doMerge(arr, l, m, r);
        }
    }
    public static void doCountingSort(int[] arr){
        int k = 0;
        for (int i = 0; i < arr.length; i++) {
            k = Math.max(k, arr[i]);
        }
        int[] count = new int[k + 1];
        int n = arr.length;
        for (int i = 0; i <= k; ++i) {
            count[i] = 0;
        }
        for (int i = 0; i < n; ++i) {
            count[arr[i]]++;
        }
        for (int i = 1; i <= k; ++i) {
            count[i] += count[i - 1];
        }
        int[] output = new int[n];
        for (int i = n - 1; i >= 0; i--) {
            output[count[arr[i]] - 1] = arr[i];
            count[arr[i]]--;
        }
        for (int i = 0; i < n; ++i) {
            arr[i] = output[i];
        }
    }
    public static int doLinearSearch(int[] arr, int x) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == x) {
                return i;
            }
        }
        return -1;
    }
    public static int doBinarySearch(int[] arr,int x){
        int l = 0, r = arr.length - 1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (arr[m] == x) {
                return m;
            }
            if (arr[m] < x) {
                l = m + 1;
            } else {
                r = m - 1;
            }
        }
        return -1;
    }
}
