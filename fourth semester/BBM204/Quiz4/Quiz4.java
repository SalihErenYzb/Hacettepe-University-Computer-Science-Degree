import java.io.*;
import java.util.*;

public class Quiz4 {
    public static void main(String[] args) throws IOException {
        
        String database = args[0];
        String query = args[1];
        BufferedReader br = new BufferedReader(new FileReader(database));
        Trie t = new Trie();
        String line;
        int count;

        // read count from the first line
        count = Integer.parseInt(br.readLine());
        for(int i=0;i<count;i++){
            line = br.readLine();
            // split by only tab not any other whitespace
            String[] parts = line.split("\t");
            // make it all lowercase
            parts[1] = parts[1].toLowerCase();
            // give runtime error if every character is not between a-z
            // for(int j=0;j<parts[1].length();j++){
            //     if(parts[1].charAt(j)<'a' || parts[1].charAt(j)>'z'){
            //         throw new RuntimeException(parts[1]+" is not a valid string");
            //     }
            // }
            t.insert(parts[1],Long.parseLong(parts[0]));
        }
        br.close();
        br = new BufferedReader(new FileReader(query));
        while((line = br.readLine())!=null){
            String[] parts = line.split("\t");
            // make it all lowercase
            parts[0] = parts[0].toLowerCase();
            /*Query received: "ba" with limit 2. Showing results:
            - 99982 bad
            - 56565 ba */
            System.out.println("Query received: \""+parts[0]+"\" with limit "+parts[1]+". Showing results:");
            ArrayList<Pair> res = t.search(parts[0],Long.parseLong(parts[1]));
            for(Pair s:res){
                System.out.println("- "+s.n.num+" "+s.s);   
            }
            if (res.size() == 0) {
                System.out.println("No results.");
            }
        }

        br.close();
    }
}
class Node{
    long num; // will be -1 if the node is not a leaf node else it will be cost of the node
    HashMap<Integer,Node> node;
    Node(){
        num = -1;
        node = new HashMap<Integer,Node>();
    }
}
class Pair implements Comparable<Pair>{
    String s;
    Node n;
    Pair(String s,Node n){
        this.s = s;
        this.n = n;
    }
    public int compareTo(Pair p){
        if(this.n.num<p.n.num){
            return -1;
        }
        else if(this.n.num>p.n.num){
            return 1;
        }
        return 0;
    }
}
class Trie{
    Node root;
    Trie(){
        root = new Node();
    }
    void insert(String s,long cost){
        Node temp = root;
        for(int i=0;i<s.length();i++){
            int index = s.charAt(i);
            // check if the node is in the hashmap
            if(temp.node.get(index)==null){
                temp.node.put(index,new Node());
            }
            temp = temp.node.get(index);
        }
        temp.num = cost;
    }
    // search takes a string and number limit, it returns all strings in trie that 
    // starts with the given string and returns all of them in sorted order of cost
    // returns at most limit number of strings
    ArrayList<Pair> search(String s,long limit){
        ArrayList<Pair> ans = new ArrayList<Pair>();
        Node temp = root;
        for(int i=0;i<s.length();i++){
            int index = s.charAt(i);
            if(temp.node.get(index)==null){
                return ans;
            }
            temp = temp.node.get(index);
        }
        // iterative dfs
        Stack<Pair> st = new Stack<Pair>();
        st.push(new Pair(s,temp));
        while(!st.isEmpty()){
            Pair p = st.pop();
            if(p.n.num!=-1){
                ans.add(p);
            }
            for(int i=0;i<500;i++){
                // if(p.n.node[i]!=null){
                //     st.push(new Pair(p.s+(char)(i),p.n.node[i]));
                // }
                if(p.n.node.get(i)!=null){
                    st.push(new Pair(p.s+(char)(i),p.n.node.get(i)));
                }
            }
        }

        Collections.sort(ans,Collections.reverseOrder());
        ArrayList<Pair> res = new ArrayList<Pair>();
        for(int i=0;i<Math.min(limit,ans.size());i++){
            res.add(ans.get(i));
        }
        return res;
    }
}