`timescale 1ns/10ps

module multiplier_tb;

	// Your code goes here.  DO NOT change anything that is already given! Otherwise, you will not be able to pass the tests!
	reg [2:0] A, B;
	wire [5:0] P;
	multiplier multiplier_inst(.A(A), .B(B), .P(P));
	integer i, j;

	initial begin
		for (i = 0; i < 8; i = i + 1) begin
			for (j = 0; j < 8; j = j + 1) begin
				A = i;
				B = j;
				#10;
			end
		end
		$finish;
	end
endmodule
