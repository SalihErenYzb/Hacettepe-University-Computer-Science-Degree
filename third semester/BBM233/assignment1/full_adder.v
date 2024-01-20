`timescale 1ns/10ps
module full_adder(
    input A, B, Cin,
    output S, Cout
);

	// Your code goes here.  DO NOT change anything that is already given! Otherwise, you will not be able to pass the tests!
    wire S0, Cout0, Cout1;
    half_adder tmp0(.A(A), .B(B), .S(S0), .C(Cout0));
    half_adder tmp1(.A(S0), .B(Cin), .S(S), .C(Cout1));
    or or0(Cout, Cout0, Cout1);

endmodule