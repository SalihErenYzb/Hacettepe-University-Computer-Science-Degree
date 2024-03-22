import java.util.*;
import java.io.*;

public class Quiz1 {
    public static void main(String[] args) throws IOException {
        
        FileReader fileReader = new FileReader(args[0]);
        ArrayList<String> toIgnore = new ArrayList<String>();
        String line = fileReader.readLine();
        while (!line.equals("...")) {
            line = line.toLowerCase(Locale.ENGLISH);
            toIgnore.add(line);
            line = fileReader.readLine();
        }
        ArrayList<String> Sentences = new ArrayList<String>();
        int index = 0;
        ArrayList<Node> keyWords = new ArrayList<Node>();
        while (fileReader.hasNextLine()) {
            line = fileReader.readLine();
            // turn line lowercase using britich locale to avoid issues with turkish i
            line = line.toLowerCase(Locale.ENGLISH);
            // split line into words
            ArrayList<String> words = new ArrayList<String>(Arrays.asList(line.split("\\s+")));
            for (int i = 0; i < words.size(); i++) {
                if (!toIgnore.contains(words.get(i))) {
                    keyWords.add(new Node(words.get(i), index));
                }
            }
            Sentences.add(line);
            // add words to sentence
            index++;    
        }
        // sort keyWords using insertion sort
        for (int i = 1; i < keyWords.size(); i++) {
            Node temp = keyWords.get(i);
            int j = i - 1;
            while (j >= 0 && keyWords.get(j).compareTo(temp) > 0) {
                keyWords.set(j + 1, keyWords.get(j));
                j--;
            }
            keyWords.set(j + 1, temp);
        }
        // print sentences based on keywords
        String keyword ="";
        String sentence ="";
        int oop = 0;
        for ( int i = 0; i < keyWords.size(); i++) {
            if ( keyword.equals(keyWords.get(i).word) && sentence.equals(Sentences.get(keyWords.get(i).index))) {
                oop++;
            }else{
                oop = 0;
            }
            keyword = keyWords.get(i).word;
            sentence = Sentences.get(keyWords.get(i).index);
            // uppercase the keyword in sentence
            // first turn sentence into array of words
            ArrayList<String> words = new ArrayList<String>(Arrays.asList(sentence.split("\\s+")));
            int k = 0;
            for (int j = 0; j < words.size(); j++) {
                if (words.get(j).equals(keyword)) {
                    if (k == oop){
                        words.set(j, words.get(j).toUpperCase(Locale.ENGLISH));
                        break;
                    }
                    k++;
                }
            }
            // join words back into sentence
            String newSentence = String.join(" ", words);
            // turn words lowercase using britich locale to avoid issues with turkish i
            for (int j = 0; j < words.size(); j++) {
                words.set(j, words.get(j).toLowerCase(Locale.ENGLISH));
            }
            System.out.println(newSentence);
        }
}
}
class Node{
    String word;
    int index;
    public Node(String word, int index){
        this.word = word;
        this.index = index;
    }
    // create a compareTo method based on just word
    public int compareTo(Node other){
        return this.word.compareTo(other.word);
    }
}
class FileReader {
    private Scanner scanner;
    public FileReader(String filename) throws IOException {
        scanner = new Scanner(new File(filename));
    }
    public String readLine() {
        return scanner.nextLine();
    }
    public boolean hasNextLine() {
        return scanner.hasNextLine();
    }
}