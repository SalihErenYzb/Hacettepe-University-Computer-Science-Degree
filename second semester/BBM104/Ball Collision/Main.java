import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.*;

class nodeForBonus{
 // this node is to store information for bonus part.
 // a single node stores a coordinate of a possible route's point 
 // it is kind of like linked list it has a variable named back which stores..
 //.. previous coordinate in a given route.
 
 // coord variable stores coordinate as expected
 // isit variable is a boolean array with size n , n donates the amount of
 // Y or R in map and it is all false at the start but as route passes thorugh 
 // kth R or Y isit[k] becomes true and when all isit is true it means
 // given route has passes all R and Y's 
 nodeForBonus back;
 int[] coord;
 boolean[] isit;
 public nodeForBonus(int n,int[] coord) {
   this.coord = coord;
   this.isit = new boolean[n];
 }
}
class FileOutput {
 /**
  * This function writes given content to file at given path.
  *
  * @param path    Path for the file content is going to be written.
  * @param content Content that is going to be written to file.
  * @param append  Append status, true if wanted to append to file if it exists, false if wanted to create file from zero.
  * @param newLine True if wanted to append a new line after content, false if vice versa.
  */
 
 public static void writeToFile(String path, String content, boolean append, boolean newLine) {
     PrintStream ps = null;
     try {
         ps = new PrintStream(new FileOutputStream(path, append));
         ps.print(content + (newLine ? "\n" : ""));
     } catch (FileNotFoundException e) {
         e.printStackTrace();
     } finally {
         if (ps != null) { //Flushes all the content and closes the stream if it has been successfully created.
             ps.flush();
             ps.close();
         }
     }
 }
}
class FileInput {
    /**
     * Reads the file at the given path and returns contents of it in a string array.
     *
     * @param path              Path to the file that is going to be read.
     * @param discardEmptyLines If true, discards empty lines with respect to trim; else, it takes all the lines from the file.
     * @param trim              Trim status; if true, trims (strip in Python) each line; else, it leaves each line as-is.
     * @return Contents of the file as a string array, returns null if there is not such a file or this program does not have sufficient permissions to read that file.
     */
    public static String[] readFile(String path, boolean discardEmptyLines, boolean trim) {
        try {
            List<String> lines = Files.readAllLines(Paths.get(path)); //Gets the content of file to the list.
            if (discardEmptyLines) { //Removes the lines that are empty with respect to trim.
                lines.removeIf(line -> line.trim().equals(""));
            }
            if (trim) { //Trims each line.
                lines.replaceAll(String::trim);
            }
            return lines.toArray(new String[0]);
        } catch (IOException e) { //Returns null if there is no such a file.
            e.printStackTrace();
            return null;
        }
    }
}
public class Main {
 
 public static String[][] boardLoader(){
  // this function reads the file and turns it into 2D String Array
  String[] input = FileInput.readFile("board.txt",true,true); 
  
  // this tmp and lengthInput variables are used to determine length of the array
  String tmp = input[0]; 
  int lengthInput = input.length; 
  int tmpInt =  tmp.split(" ",-2).length;  
  
  //our board array
  String[][] board = new String[ lengthInput ][ tmpInt ];
  
  
  // building board by splitting the input
  for (int i = 0 ; i < lengthInput ; i++) {
   board[i] = input[i].split(" ",-2);
  }
  return(board);
 }
 
 // reading moves
 static String[] moveLoader() {
  String[] input = FileInput.readFile("move.txt",true,true); 
  
  return( input[0].split(" ",-2) );
 }
 
 
 static int[] coordinateFinder(String[][] board) {
  
  // this whole function finds "*" in the board
  // if it can't find it it return {-1,-1} array
  
  int[] tmp = {-1,-1};  
  String tempo = "*"; 
  
  int z = board.length; 
  int y = board[0].length; 
  
  
  
  
  for (int i = 0 ; i < z ; i++) {
   
   for (int j = 0 ; j < y ; j++) {
    
    
    if (board[i][j].equals(tempo)) {// if it equals "*"
     tmp[0] = i;
     tmp[1] = j;
    }
   }
  }
  return(tmp);
 }
 
 
 public static int coordinateChanger(int i,int length) {
  
  // this functions is very important
  //given a new coordinate and it's dimension's length it converts the coordinate...
  //...to boundaries of given length
  //this makes it so left of leftist square is rightest square
  if ((0 <= i) && (i < length)) {
   return(i);
   
  }
  else if (i == length) {// if you are in boundary you go to start
   return(0);
  }
  else  {
   return(length-1);// if you are in 0th square and go left you end up in rightes
  }
 }
 static int xFinder(String i) {// this converst moves into changes they represent in x dimension
  int xIncrement;
  if (i.equals("L")) {
   xIncrement = 0;
  }
  else if (i.equals("R")) {
   xIncrement = 0;
  }
  else if (i.equals("U")) {
   xIncrement = -1;
  }
  else{
   xIncrement = 1;
  }
  return(xIncrement);
 }
 static int yFinder(String i) {// this converst moves into changes they represent in y dimension
  int yIncrement;
  if (i.equals("L")) {
   yIncrement = -1;
  }
  else if (i.equals("R")) {
   yIncrement = 1;
  }
  else if (i.equals("U")) {
   yIncrement = 0;
  }
  else{
   yIncrement = 0;
  }
  return(yIncrement);
 }
 static void boardPrinter(String[][] board,String path) {// this function prints board
  for (String[] i : board) {
   //this nested for adds every element in board to a String and writes it to file
   String tmp = "";
   for (String j : i) {
    tmp += j + " "; 
   }
   // reason we use substring is to make sure last " " is not involved
   FileOutput.writeToFile(path, tmp.substring(0, tmp.length() - 1 ), true, true);
  }
  FileOutput.writeToFile(path, "", true, true);// adds /n
 }
public static int findIndex(int[][] c, int[] coord) {
 // this function finds a given coordinate in a array of coordinates
 // and returns the index of that element
 // if element does not exist it will return -1
 // this f is for bonus part
 
  for (int i = 0; i < c.length; i++) {
      if (Arrays.equals(c[i], coord)) {
          return i;
      }
  }

  return -1;
}
static String[] getMovesForBonus(String[][] board) {
 
 //this function will find smallest path that goes thorough all R and Y 's
 int tmp = 0;
 
 int z = board.length; 
 int y = board[0].length; 
 // first it starts by finding all R's and Y's
 // this for is intended to only find the number of Rs and Ys
 for (int i = 0 ; i < z ; i++) {
  
  for (int j = 0 ; j < y ; j++) {
   
   
   if ((board[i][j].equals("Y")) || (board[i][j].equals("R"))){
    tmp++;
   }
  }
 }
 //we create allCoords variable to store all Y and R
 int[][] allCoords = new int[tmp][2];
 int m = 0;
 for (int i = 0 ; i < z ; i++) {
  
  for (int j = 0 ; j < y ; j++) {
   
   
   if ((board[i][j].equals("R")) || (board[i][j].equals("Y"))){
    allCoords[m][0] = i;
    allCoords[m][1] = j;
    m++;
   }
  }
 }
 //after getting all coordinates
 // we create a queue and start a bread first search algorithm
 
 Queue<nodeForBonus> stack = new LinkedList<nodeForBonus>();
 nodeForBonus start = new nodeForBonus(tmp ,coordinateFinder(board) );
 // first it creates a node of Starting position and puts it in queue
 stack.add(start);
 nodeForBonus tmpNode = null;//this node denotes every new node that gets created
 int calan = 0;// this variable is zero as long as no routes reach all r and y
 while ((stack.size() != 0) && (calan == 0)){
  
  nodeForBonus stackTmp = stack.remove();
  // this stackTmp node denotes the last route in queue and 
  // we will add all moves this variable can do to queue as a node

  int[] tempC = stackTmp.coord;
  // this possibleMoves has 4 moves that stackTmp can do up, left,right, down
  // it makes sure moves are in right length using coordinateChanger
  int[][] possibleMoves = {{coordinateChanger(tempC[0]+1,z),coordinateChanger(tempC[1],y)}
                             ,{coordinateChanger(tempC[0]-1,z),coordinateChanger(tempC[1],y)}
                               ,{coordinateChanger(tempC[0],z),coordinateChanger(tempC[1]+1,y)}
                                 ,{coordinateChanger(tempC[0],z),coordinateChanger(tempC[1]-1,y)}};
  
  
  for (int[] elementC : possibleMoves) {
   String tmpString = board[elementC[0]][elementC[1]];// element possible move has
   
   // if element is a wall or a hole it passes it
   if ((!tmpString.equals("W")) && (!tmpString.equals("H"))) {
    // for legal moves it creates a new node with isit array of previous node
    
    tmpNode = new nodeForBonus(tmp , elementC.clone());
    tmpNode.back = stackTmp;
    tmpNode.isit = tmpNode.back.isit.clone();
    
    // this line looks if particular possible moves is R or Y if so finds it's index in isit array
    int index = findIndex(allCoords , elementC);
    
    // this lines first updates new node's isit only if index is legal then it checks
    // if isit is full of trues values
    // if so it increases the variable calan to halt the loop 
    //and find the route that goes thoruogh every R and Y
    if (index >= 0) {
     tmpNode.isit[index] = true;
     boolean test = false;
     for (boolean element : tmpNode.isit) {
         if (!element) {
             test = true;
             break;
         }

     
    }
     if (!test) {// increasing of calan in case it is all true

      calan++;
      break;
     }
    }
    stack.add(tmpNode);// if it is not all true it just adds the node to queue
    
   }
  }
 }
  
  
  // now we got a node named tmpNode that has the route we need we have to extract it
  Stack<String> allMoves = new Stack<String>();
  // we create a stack to get them
  int number = 0;
  do  {
   // this loop turns the coordinates of node queue to D , L , R, U
   // by looking at the difference between node and previous node
   
   number++;
   String a = "";
   int tempInt = tmpNode.coord[0]-tmpNode.back.coord[0];
   int tempInt2 = tmpNode.coord[1]-tmpNode.back.coord[1];
   if ( ( tempInt == 1) || ( tempInt == 1-z) ){
    a = "D";
   }
   else if (( tempInt == -1) || ( tempInt == z-1) ){
    a = "U";
   }
   else if (( tempInt2 == -1) || ( tempInt2 == y-1)) {
    a = "L";
   }
   else if (( tempInt2== 1) || ( tempInt2 == 1-y)) {
    a = "R";
   }
   allMoves.push(a);
   tmpNode = tmpNode.back;
  }while (tmpNode.back != null);
  
  // now we got a stack of moves we have to turn them into an array
  String[] allMove = new String[number]; // here is our arrray
  int r = 0;
  // this loop fills the array using stack
  while (!allMoves.isEmpty()) {
   allMove[r] = allMoves.pop();
   r++;
  }
  return(allMove);
 
 
 
}

static void output(String path) {// in case someone uses it without giving it the normal command
  output("normal",path);
 }
static void output(String state, String path) {
 // this function given path and and normal or bonus for 2 different mods writes a file
 
int totalScore = 0;//score
String[] moves = {"null"};
String[][] board = Main.boardLoader();//uses the functions we already wrote

if (state.equals("normal")) {// this basic if will get moves depending on game mode
moves = Main.moveLoader();
} else if (state.equals("bonus")) {
moves = Main.getMovesForBonus(board);
}

int lengthX = board.length, lengthY = board[0].length;// theese length are needed always

int[] coords = Main.coordinateFinder(board);// this is the coordinate of "*"
int xCoord = coords[0], yCoord = coords[1];//turn it into x and y

int tempXCoord = 0, tempYCoord = 0; // will be used while iterating thorough moves

FileOutput.writeToFile(path, "Game board:", true, true);//writing initial state of board
boardPrinter(board, path);



boolean gameOver = false;
for (int index = 0 ; index < moves.length ; index++ ) {
    String i = moves[index];
    tempXCoord = coordinateChanger(xCoord + xFinder(i), lengthX);
    // new x and y coordinates according to move
    tempYCoord = coordinateChanger(yCoord + yFinder(i), lengthY);

    String tmp = board[tempXCoord][tempYCoord];// this holds the String in the new coord

    if (tmp.equals("H")) {
        board[xCoord][yCoord] = " ";
        gameOver = true;
        moves = Arrays.copyOfRange(moves, 0, index+1); 
        // it makes sure code does not write extra moves
        break;// when it fall into hole it stops
    } else if (tmp.equals("B")) {
        totalScore -= 5;
        board[xCoord][yCoord] = "X";
    } else if (tmp.equals("R")) {
        totalScore += 10;
        board[xCoord][yCoord] = "X";
    } else if (tmp.equals("Y")) {
        totalScore += 5;
        board[xCoord][yCoord] = "X";
    } else if (tmp.equals("W")) {
       // when it is a wall it has to find new coordinates
       // and do the same procedure that it did
        tempXCoord = coordinateChanger(xCoord - xFinder(i), lengthX);
        tempYCoord = coordinateChanger(yCoord - yFinder(i), lengthY);
        tmp = board[tempXCoord][tempYCoord];
        if (tmp.equals("H")) {
            board[xCoord][yCoord] = " ";
            gameOver = true;
            moves = Arrays.copyOfRange(moves, 0, index+1); 
            break;
        } else if (tmp.equals("B")) {
            totalScore -= 5;
            board[xCoord][yCoord] = "X";
        } else if (tmp.equals("R")) {
            totalScore += 10;
            board[xCoord][yCoord] = "X";
        } else if (tmp.equals("Y")) {
            totalScore += 5;
            board[xCoord][yCoord] = "X";
        } else {
         // if our player goes to a place which is R or Y it has to put 
         //  X in there but if it is not it flips the values of before and after indexes
            board[xCoord][yCoord] = board[tempXCoord][tempYCoord];
        }
    } else {
     // if our player goes to a place which is R or Y it has to put 
     //  X in there but if it is not it flips the values of before and after indexes
        String tmpVar = board[tempXCoord][tempYCoord];
        board[xCoord][yCoord] = tmpVar;
    }
    //our new player
    board[tempXCoord][tempYCoord] = "*";

    xCoord = tempXCoord;// our new coordinates
    yCoord = tempYCoord;

}
String tmp1 = "";// finds the String of moves to write
for (String j : moves) {
    tmp1 += j + " ";
}

if (state.equals("bonus")) {// writes necessery info
 FileOutput.writeToFile(path, "Your optimal path is:", true, true);
 FileOutput.writeToFile(path, tmp1.substring(0, tmp1.length() - 1) +"\n", true, true);
 FileOutput.writeToFile(path, "Number of movements: " + moves.length+"\n" , true, true);



}else if (state.equals("normal")) {// writes necessery info
 FileOutput.writeToFile(path, "Your movement is:", true, true);
 FileOutput.writeToFile(path, tmp1.substring(0, tmp1.length() - 1)+"\n" , true, true);
}


FileOutput.writeToFile(path, "Your output is:", true, true);

boardPrinter(board, path);// writes new board
if (gameOver) {
 FileOutput.writeToFile(path, "Game Over!", true, true);
}

if (state.equals("bonus")) {
 FileOutput.writeToFile(path, "Best score: " + totalScore, true, true);
}
else {

FileOutput.writeToFile(path, "Score: " + totalScore , true, true);
}
}
 public static void main(String[] args) {


  output("normal","output.txt");//code automatically works for normal and bonus
  output("bonus","bonus.txt");
 }
}
