
public class Book {
    private static int id1 = 1;
    private int id ;
    private boolean isHandwritten;

    public Book(boolean isHandwritten) {
        this.id = Book.id1;
    	this.isHandwritten = isHandwritten;
    	id1++;
    }
    public boolean getIsHandWritten() {
    	return isHandwritten;
    }
    public int getId() {
        return id;
    }
    public String longVersion() {
    	if (this.isHandwritten==true) {
    		return "Handwritten";
    	}else {
    		return "Printed";
    	}
    }

    public boolean isHandwritten() {
        return isHandwritten;
    }
}