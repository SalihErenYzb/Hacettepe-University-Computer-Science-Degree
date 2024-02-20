import java.time.LocalDate;
import java.time.format.DateTimeFormatter;

public class ReadInLoan {
	private int memberId;
	private int bookId;
	private LocalDate borrowDate;
	ReadInLoan(int member,int book,LocalDate date){
		this.setMemberId(member);
		this.setBookId(book);
		this.setBorrowDate(date);
	}
	public int getMemberId() {
		return memberId;
	}
	public void setMemberId(int memberId) {
		this.memberId = memberId;
	}
	public int getBookId() {
		return bookId;
	}
	
	public String toString() {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");
        String dateString = getBorrowDate().format(formatter);
		return "The book ["+getBookId()+"] was borrowed by member ["+getMemberId()+"] at "+dateString;
	}
	public String toString1() {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");
        String dateString = getBorrowDate().format(formatter);
		return "The book ["+getBookId()+"] was read in library by member ["+getMemberId()+"] at "+dateString;
	}
	public void setBookId(int bookId) {
		this.bookId = bookId;
	}
	public LocalDate getBorrowDate() {
		return borrowDate;
	}
	public void setBorrowDate(LocalDate borrowDate) {
		this.borrowDate = borrowDate;
	}
	
}
