import java.time.temporal.ChronoUnit;

import java.util.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
class Library {
    public int[] infoArray = new int[4]; 
    private Map<Integer, Book> idToBook;
    private Map<Integer, Member> idToMember;
    private Map<Integer, Loan> idToLoan;
    private Map<Integer, ReadInLoan> idToReadInLoan;

    public Library() {
    	for (int i = 0; i<4 ;i++) {
    		infoArray[i] = 0;
    	}
        idToBook = new HashMap<>();
        idToMember = new HashMap<>();
        idToLoan = new HashMap<>();
        idToReadInLoan = new HashMap<>();

    }

	/**
	
	Allows a member to read a book in the library.
	
	@param input an array of String containing the book id,
	 member id, and the date the book is read in yyyy-MM-dd format.
	
	@throws IllegalArgumentException if the member or book does not exist, 
	the book is already being read,
	 the book is handwritten and the member is a student, 
	 or the input is in an invalid format.
	*/
    public void readInLibrary(String[] input) {
    	try {
	    	int idB = Integer.parseInt(input[1]);
	    	int idM = Integer.parseInt(input[2]);
	    	String tmp2 = input[3];
	    	LocalDate date = LocalDate.parse(input[3]);
	    	boolean dataCheck = this.isTrue(idB, idM);
	    	if (!dataCheck) {
	    		throw new IllegalArgumentException("You can not read this book!");
	    	}
	    	dataCheck = dataCheck && !this.isItBeingRead(idB);
	    	if (!dataCheck) {
	    		throw new IllegalArgumentException("You can not read this book!");
	    	}

	    	dataCheck = dataCheck && this.canReadThisInLib(idB, idM);

	    	if (dataCheck) {
	    		ReadInLoan loan = new ReadInLoan(idM, idB, date);
	    		this.setIdToReadInLoan(idB, loan);
	    		FileOutput.writeToFile(Main.opt,"The book ["+idB+"] was read in library by member ["+
	    		idM+"] at "+tmp2, true, true);
	    	}else {
	    		throw new IllegalArgumentException("You can not read this book!");
	    	}
    	}catch (Exception e) {
    		throw new IllegalArgumentException(e.getMessage());
    	}
    }
    /**

    Determines whether a member with the given ID can read a book with the given ID.
    If the book is handwritten and the member is a student, throws an IllegalArgumentException
    with the message "Students can not read handwritten books!". Otherwise, returns true.
    @param idB the ID of the book
    @param idM the ID of the member
    @return true if the member can read the book, false otherwise
    */
    public boolean canReadThisInLib(int idB, int idM) {
    	if (idToBook.get(idB).getIsHandWritten() && ( idToMember.get(idM).isStudent() == true )){
    		throw new IllegalArgumentException("Students can not read handwritten books!");
    	}
    	else {
    		return true;
    	}
    }

	/**
	
	This method extends the deadline for a book borrowed by a member
	
	@param input String array containing the input parameters: book ID, member ID, and new deadline date
	
	@throws IllegalArgumentException if the input parameters are invalid or the book cannot have its deadline extended
	*/
    public void extendBook(String[] input) {
    	try {
	    	int idB = Integer.parseInt(input[1]);
	    	int idM = Integer.parseInt(input[2]);
	    	LocalDate date = LocalDate.parse(input[3]);

	    	boolean dataCheck = this.isTrue(idB, idM);
	    	if (!dataCheck) {
	    		throw new IllegalArgumentException("You cannot extend the deadline!");
	    	}
	    	dataCheck = dataCheck && didMemberBorrowThisBookWithLoan(idB, idM);
	    	if (!dataCheck) {
	    		throw new IllegalArgumentException("You cannot extend the deadline!");
	    	}

	    	dataCheck = dataCheck && this.isLoanExtandable(idB);
	    	if (!dataCheck) {
	    		throw new IllegalArgumentException("You cannot extend the deadline!");
	    	}
	    	dataCheck = dataCheck && !this.isItPastDeadLine(idB, date);
	    	if (!dataCheck) {
	    		throw new IllegalArgumentException("You cannot extend the deadline!");
	    	}
	    	dataCheck = dataCheck && !this.isHandWritten(idB);
	    	if (!dataCheck) {
	    		throw new IllegalArgumentException("You cannot extend the deadline!");
	    	}
    	
	    	if (dataCheck) {
	    		Loan loan2 = idToLoan.get(idB);
	            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");
	            String dateString = loan2.getDeadline().format(formatter);

	    		FileOutput.writeToFile(Main.opt,
	    					 "The deadline of book ["+idB+
	    					 "] was extended by member ["+idM+"] at "+input[3],true,true);
	    		loan2.extend();

	    		this.setIdToLoan(idB, loan2);
	    		dateString = loan2.getDeadline().format(formatter);
	    		FileOutput.writeToFile(Main.opt,
	    			"New deadline of book ["+idB+"] is "+dateString,true,true);

	    	}else {
	    		throw new IllegalArgumentException("You cannot extend the deadline!");
	    	}
    	}catch (Exception e) {
    		throw new IllegalArgumentException("You cannot extend the deadline!");
    	}
    }

	/**
	
	Borrows a book to a member if the book is available, the member can borrow this book,
	and the member did not exceed the maximum number of books allowed to borrow.
	@param input An array of strings containing input parameters such as book id, member id, and date.
	@throws IllegalArgumentException if the book is not available, the member cannot borrow this book,

	    or the member has exceeded the maximum number of books allowed to borrow.
	*/
    public void borrowBook(String[] input) {
    	try {
	    	int idB = Integer.parseInt(input[1]);
	    	int idM = Integer.parseInt(input[2]);
	    	String tmp2 = input[3];
	    	LocalDate date = LocalDate.parse(input[3]);
	    	boolean dataCheck = this.isTrue(idB, idM);
	    	if (!dataCheck) {
	    		throw new IllegalArgumentException("You can not read this book!");
	    	}
	    	
	    	dataCheck = dataCheck && !this.isItBeingRead(idB);
	    	if (!dataCheck) {
	    		throw new IllegalArgumentException("You can not read this book!");
	    	}
	    	
	    	dataCheck = dataCheck && this.canMemBorrowThisBook(idB, idM);
	    	if(!this.canMemberRead(idM)) {
	    		throw new IllegalArgumentException("You have exceeded the borrowing limit!");
	    	}

	    	if (dataCheck) {
	    		Loan loan = new Loan(idM, idB, date, idToMember.get(idM).isStudent());
	    		this.setIdToLoan(idB, loan);
	    		idToMember.get(idM).setLoanCount(idToMember.get(idM).getLoanCount()+1);
	    		FileOutput.writeToFile(Main.opt,"The book ["+idB+"] was borrowed by member ["+
	    		idM+"] at "+tmp2, true, true);
	    	}else {
	    		throw new IllegalArgumentException("You can not read this book!");
	    	}
    	}catch (Exception e) {
    		throw new IllegalArgumentException(e.getMessage());
    	}
    }
    /**
     * Adds a new book to the library.
     *
     * @param input an array of input strings containing information about the book to be added
     *              The first element of the array is the book type, which can be either "H" for handwritten
     *              or "P" for printed. The remaining elements are not used.
     */
    public void addBook(String[] input) {
        String memtype = input[1];
        Book tmp;
        if (memtype.equals("H")) {
            infoArray[3]++;
            tmp = new Book(true);
            idToBook.put(tmp.getId(), tmp);
            FileOutput.writeToFile(Main.opt,"Created new book: Handwritten [id: "
            + tmp.getId() + "]",true,true);
        } else if (memtype.equals("P")) {
            infoArray[2]++;
            tmp = new Book(false);
            idToBook.put(tmp.getId(), tmp);
            FileOutput.writeToFile(Main.opt,"Created new book: Printed [id: "
            + tmp.getId() + "]",true,true);
        }
    }

	/**
	
	Adds a new member to the library.
	@param input an array of input strings containing member type information
	@throws IllegalArgumentException if the member type is invalid
	*/
    public void addMember(String[] input) {
        String memtype = input[1];
        Member tmp;
        if (memtype.equals("S")) {
            infoArray[0]++;
            tmp = new Member(true);
            idToMember.put(tmp.getId(), tmp);
            FileOutput.writeToFile(Main.opt,"Created new member: Student [id: " 
            + tmp.getId() + "]",true,true);
        } else if (memtype.equals("A")) {
            infoArray[1]++;
            tmp = new Member(false);
            idToMember.put(tmp.getId(), tmp);
            FileOutput.writeToFile(Main.opt,"Created new member: Academic [id: " 
            + tmp.getId() + "]",true,true);
        }
    }
    /**

    Determines whether a loan with the given book ID is past its deadline, given the current date.
    Retrieves the loan object associated with the book ID, and compares the given date to the loan's deadline.
    Returns true if the given date is after the loan's deadline, and false otherwise.
    @param idB the ID of the book associated with the loan
    @param date the current date
    @return true if the loan is past its deadline, false otherwise
    */
    public boolean isItPastDeadLine(int idB, LocalDate date) {
    	Loan loan = idToLoan.get(idB);

    	return date.isAfter(loan.getDeadline());
    }

	/**
	
	Determines whether a loan with the given ID is extendable.
	Retrieves the loan object associated with the given ID, and returns
	the value of its "isExtandable" property.
	@param id the ID of the loan
	@return true if the loan is extendable, false otherwise
	*/
    public boolean isLoanExtandable(int id) {
    	Loan loan = idToLoan.get(id);
    	return loan.isExtandable();
    }
    /**

    Determines whether a member with the given ID borrowed a book with the given ID
    and read it during the loan.
    Checks if the given book ID is associated with a ReadInLoan object in the idToReadInLoan map.
    If not, returns false. Otherwise, retrieves the ReadInLoan object and checks if its member ID
    matches the given member ID. Returns true if the member ID matches, and false otherwise.
    @param idB the ID of the book
    @param idM the ID of the member
    @return true if the member borrowed the book and read it during the loan, false otherwise
    */
    public boolean didMemberBorrowThisBookWithReadInLoan(int idB, int idM) {
    	if (!idToReadInLoan.containsKey(idB)) {return false;}
    	ReadInLoan c = idToReadInLoan.get(idB);
    	if (c == null) {return false;}
    	if (c.getMemberId() == idM) {return true;}
    	return false;
    }
    /**

    Determines whether a member with the given ID borrowed a book with the given ID.
    Checks if the given book ID is associated with a Loan object in the idToLoan map.
    If not, returns false. Otherwise, retrieves the Loan object and checks if its member ID
    matches the given member ID. Returns true if the member ID matches, and false otherwise.
    @param idB the ID of the book
    @param idM the ID of the member
    @return true if the member borrowed the book, false otherwise
    */
    public boolean didMemberBorrowThisBookWithLoan(int idB, int idM) {
    	if (!idToLoan.containsKey(idB)) {return false;}
    	Loan c = idToLoan.get(idB);
    	if (c == null) {return false;}
    	if (c.getMemberId() == idM) {return true;}
    	return false;
    }

	/**
	
	Determines whether a member with the given ID can borrow a book with the given ID.
	Retrieves the Book object associated with the given book ID, and checks if it is handwritten.
	If the book is handwritten, returns false, as handwritten books cannot be borrowed.
	Otherwise, returns true, as the member can borrow the book.
	@param idB the ID of the book
	@param idM the ID of the member
	@return true if the member can borrow the book, false otherwise
	*/
    public boolean canMemBorrowThisBook(int idB, int idM) {
    	Book book = idToBook.get(idB);
    	if (book.getIsHandWritten()) {return false;}

	    else {return true;}
    	
    }

	/**
	
	Determines whether a member with the given ID can read books.
	Retrieves the Member object associated with the given ID, and checks if the member can borrow more books.
	If the member can borrow more books, returns true, as the member can read.
	Otherwise, returns false, as the member has reached their borrowing limit and cannot read more books.
	@param id the ID of the member
	@return true if the member can read, false otherwise
	*/
    public boolean canMemberRead(int id) {
    	Member mem = idToMember.get(id);
    	return mem.canBorrowMore();
    }

	/**
	
	Determines whether a book with the given ID is currently being read by a member.
	Checks if the given book ID is associated with either a Loan or a ReadInLoan object in the idToLoan or idToReadInLoan map.
	If neither is found, returns false, as the book is not being read. Otherwise, returns true.
	@param id the ID of the book
	@return true if the book is being read, false otherwise
	*/
    public boolean isItBeingRead(int id) {
    	if ((!idToLoan.containsKey(id) || idToLoan.get(id)==null) &&
    			(!idToReadInLoan.containsKey(id) || idToReadInLoan.get(id)==null)) {
    		return false;
    	}
    	else { return true;}
    }

	/**
	
	Determines whether a book with the given ID is handwritten.
	Retrieves the Book object associated with the given ID, and checks if it is handwritten.
	If the book is handwritten, returns true. Otherwise, returns false.
	@param id the ID of the book
	@return true if the book is handwritten, false otherwise
	*/
    public boolean isHandWritten(int id) {
    	Book tmp = idToBook.get(id);
    	if (tmp.isHandwritten()) {
    		return true;
    	}
    	else { return false;}
    }

	/**
	
	Determines whether a book with the given ID exists in the library's catalog.
	Checks if the given book ID is associated with a Book object in the idToBook map.
	If the book exists, returns true. Otherwise, returns false.
	@param id the ID of the book
	@return true if the book exists, false otherwise
	*/
    public boolean doesBookExist(int id) {
    	if (idToBook.containsKey(id)) {return true;}
    	else { return false;}
    }
    /**

    Determines whether a member with the given ID exists in the library's membership records.
    Checks if the given member ID is associated with a Member object in the idToMember map.
    If the member exists, returns true. Otherwise, returns false.
    @param id the ID of the member
    @return true if the member exists, false otherwise
    */
    public boolean doesMemberExist(int id) {
    	if (idToMember.containsKey(id)) {return true;}
    	else { return false;}
    }
    /**

    Determines whether both a book with the given book ID and a member with the given member ID exist in their respective records.
    Calls the doesMemberExist() and doesBookExist() methods to check if the given member ID and book ID exist, respectively.
    If both the member and book exist, returns true. Otherwise, returns false.
    @param idB the ID of the book
    @param idM the ID of the member
    @return true if both the book and member exist, false otherwise
    */
    public boolean isTrue(int idB, int idM) {
    	return (doesMemberExist(idM) && doesBookExist(idB));
    }
	public Book getidToBook(int id) {
		return idToBook.get(id);
	}

	public void setidToBook(Integer id,Book book) {
		idToBook.put(id,book ) ;
	}
	public Member getidToMember(int id) {
		return idToMember.get(id);
	}

	public void setidToMember(Integer id,Member member) {
		idToMember.put(id,member ) ;
	}
	public ReadInLoan getIdToLoan(int id) {
		return idToLoan.get(id);
	}

	public void setIdToLoan(Integer id,Loan loan) {
		idToLoan.put(id,loan ) ;
	}
	public ReadInLoan getIdToReadInLoan(int id) {
		return idToReadInLoan.get(id);
	}

	public void setIdToReadInLoan(Integer id,ReadInLoan loan) {
		idToReadInLoan.put(id,loan ) ;
	}
	/**

	Determines whether a book with the given book ID is currently either being loaned or read by a member with the given member ID.
	Checks if the book with the given ID is currently in the idToReadInLoan or idToLoan maps.
	If the book is in the idToReadInLoan map but not in the idToLoan map, returns true, indicating that the book is being read.
	If the book is in the idToLoan map but not in the idToReadInLoan map, returns false, indicating that the book is being loaned.
	If the book is not in either map, throws an IllegalArgumentException with the message "can not return the book".
	@param idB the ID of the book
	@param idM the ID of the member
	@return true if the book is being read, false if the book is being loaned, or throws an IllegalArgumentException if the book is not in either map
	*/
	public boolean loanOrReadInLoan(int idB, int idM) {
    	ReadInLoan readLoan = idToReadInLoan.get(idB);

    	Loan loan = idToLoan.get(idB);

    	if (readLoan== null  ) {
    		return true;
    	}else if (loan==null) {
    		return false;
    	}else {
    		throw new IllegalArgumentException("can not return the book");
    	}
	}

	public void returnBook(String[] input) {
    	try {
	    	int idB = Integer.parseInt(input[1]);
	    	int idM = Integer.parseInt(input[2]);
	    	String tmp2 = input[3];
	    	LocalDate date = LocalDate.parse(input[3]);
	    	boolean dataCheck = this.isTrue(idB, idM);
	    	if (!dataCheck) {
	    		throw new IllegalArgumentException("Book could not be returned");
	    	}
	    	dataCheck = dataCheck && this.isItBeingRead(idB);
	    	if (!dataCheck) {
	    		throw new IllegalArgumentException("Book could not be returned");
	    	}
    		int fee = 0;

	    	if (this.loanOrReadInLoan(idB, idM) && dataCheck) {
	    		boolean tmpbool = this.isItPastDeadLine(idB , date);
	    		
	    		if (tmpbool) {

	    	        fee = (int) ChronoUnit.DAYS.between(idToLoan.get(idB).getDeadline(), date);
	    		}

	    		idToLoan.remove(idB);

	    		idToMember.get(idM).setLoanCount(idToMember.get(idM).getLoanCount()-1);

	    		FileOutput.writeToFile(Main.opt,"The book ["+idB+
	    				"] was returned by member ["+idM+"] at "+tmp2+" Fee: "+fee,true,true);
	    	}else if (dataCheck) {

	    		idToReadInLoan.remove(idB);

	    		FileOutput.writeToFile(Main.opt,"The book ["+idB+
	    				"] was returned by member ["+idM+"] at "+tmp2+" Fee: "+fee,true,true);

	    	}else {

	    		throw new IllegalArgumentException("Book could not be returned");
	    	}
    	}catch (Exception e) {
    		throw new IllegalArgumentException(e.getMessage());
    	}
	}
	public void getTheHistory(String[] parts) {
        Map<Integer, Member> sortedMap1 = new TreeMap<>(new Comparator<Integer>() {
            public int compare(Integer key1, Integer key2) {
                return key1.compareTo(key2);
            }
        });
        sortedMap1.putAll(idToMember);

        FileOutput.writeToFile(Main.opt,
        		"History of library:\n\nNumber of students: "+infoArray[0],true,true);
        for (Map.Entry<Integer, Member> entry : sortedMap1.entrySet()) {
            if (entry.getValue().isStudent()) {
            	FileOutput.writeToFile(Main.opt,
            			"Student [id: " + entry.getKey() + "]",true,true);
            }
        }
        FileOutput.writeToFile(Main.opt, "", true, true);
        FileOutput.writeToFile(Main.opt,
        		"Number of academics: "+infoArray[1],true,true);
        for (Map.Entry<Integer, Member> entry : sortedMap1.entrySet()) {
            if (!entry.getValue().isStudent()) {
            	FileOutput.writeToFile(Main.opt,
            			"Academic [id: " + entry.getKey() + "]",true,true);
            }
        }
		
        FileOutput.writeToFile(Main.opt, "", true, true);

		
		
        Map<Integer, Book> sortedMap = new TreeMap<>(new Comparator<Integer>() {
            public int compare(Integer key1, Integer key2) {
                return key1.compareTo(key2);
            }
        });
        sortedMap.putAll(idToBook);
        FileOutput.writeToFile(Main.opt,
        		"Number of printed books: "+infoArray[2],true,true);
        for (Map.Entry<Integer, Book> entry : sortedMap.entrySet()) {
            if (!entry.getValue().isHandwritten()) {
            	FileOutput.writeToFile(Main.opt,
            			"Printed [id: " + entry.getKey() + "]",true,true);
            }
        }
        FileOutput.writeToFile(Main.opt, "", true, true);

        FileOutput.writeToFile(Main.opt,
        		"Number of handwritten books: "+infoArray[3],true,true);
        for (Map.Entry<Integer, Book> entry : sortedMap.entrySet()) {
            if (entry.getValue().isHandwritten()) {
            	FileOutput.writeToFile(Main.opt,
            			"Handwritten [id: " + entry.getKey() + "]",true,true);
            }
        }
        FileOutput.writeToFile(Main.opt, "", true, true);

        FileOutput.writeToFile(Main.opt,
        		"Number of borrowed books: "+idToLoan.size(),true,true);
        Map<Integer, Loan> sortedMap3 = new TreeMap<>(new Comparator<Integer>() {
            public int compare(Integer key1, Integer key2) {
                return key1.compareTo(key2);
            }
        });
        sortedMap3.putAll(idToLoan);
        for (Map.Entry<Integer, Loan> entry : sortedMap3.entrySet()) {
        	FileOutput.writeToFile(Main.opt,
        			entry.getValue().toString(),true,true);
            
        }
        FileOutput.writeToFile(Main.opt, "", true, true);

        FileOutput.writeToFile(Main.opt,
        		"Number of books read in library: "+idToReadInLoan.size(),true,false);
        Map<Integer, ReadInLoan> sortedMap4 = new TreeMap<>(new Comparator<Integer>() {
            public int compare(Integer key1, Integer key2) {
                return key1.compareTo(key2);
            }
        });
        sortedMap4.putAll(idToReadInLoan);
        for (Map.Entry<Integer, ReadInLoan> entry : sortedMap4.entrySet()) {
        	FileOutput.writeToFile(Main.opt,"",true,true);
        	FileOutput.writeToFile(Main.opt,
        			entry.getValue().toString1(),true,false);
            
        }
		
	}
}