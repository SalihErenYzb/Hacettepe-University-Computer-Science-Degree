
public class Member {
    private static int id1 = 1;

    private int id;
    private boolean isStudent;  // true if the member is a student, false if they are an academic
    private int loanCount;
    public Member(boolean isStudent) {
        this.id = Member.id1;
        this.isStudent = isStudent;
        this.setLoanCount(0);
        Member.id1++;
    }

    public int getId() {
        return id;
    }



    public boolean isStudent() {
        return isStudent;
    }

	public int getLoanCount() {
		return loanCount;
	}

	public void setLoanCount(int loanCount) {
		this.loanCount = loanCount;
	}
	public boolean canBorrowMore() {
		if (this.isStudent()==true && this.getLoanCount() < 2) {
			return true;
		}else if (this.isStudent()==false && this.getLoanCount() < 4) {
			return true;
		}else {
			return false;
		}
	}
}


