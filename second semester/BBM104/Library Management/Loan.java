import java.time.LocalDate;
public class Loan extends ReadInLoan{
	private LocalDate deadline;
	private boolean isExtandable;
	private boolean isStudent;
	Loan(int member,int book,LocalDate date,boolean isStudent){
		super(member,book,date);
		this.isStudent = isStudent;
		this.setExtandable(true);
		if (isStudent) {
			this.setDeadline(this.getBorrowDate().plusDays(7));
		}else {
			this.setDeadline(this.getBorrowDate().plusDays(14));

		}
	}
	public void extend() {
		if (this.isExtandable()) {
			if (isStudent) {
				this.deadline = this.deadline.plusDays(7);
			}else {
				this.deadline = this.deadline.plusDays(14);

			}
			this.setExtandable(false);
		}else {
			throw new IllegalArgumentException("You cannot extend the deadline!");
		}
	}

	public void setDeadline(LocalDate borrowDate) {
		this.deadline = borrowDate;
	}

	public LocalDate getDeadline() {
		return deadline;
	}
	public boolean isExtandable() {
		return isExtandable;
	}
	public void setExtandable(boolean isExtandable) {
		this.isExtandable = isExtandable;
	}
	public boolean isStudent() {
		return isStudent;
	}
}
