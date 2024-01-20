`timescale 1ns/10ps

module multiplier (
    input [2:0] A, B,
    output [5:0] P
);
    and and0_inst(P[0], A[0], B[0]);
    wire and1, and2, and3, and4, and5, and6, and7, and8;
    //change all to structural design
    and and1_inst(and1, A[1], B[0]);
    
    and and2_inst(and2, A[0], B[1]);
    wire c0;
    half_adder ha1(.A(and2), .B(and1), .S(P[1]), .C(c0));

    and and3_inst(and3, A[2], B[0]);
    and and4_inst(and4, A[1], B[1]);
    wire s1,c1;
    half_adder ha2(.A(and4), .B(and3), .S(s1), .C(c1));
    wire c2;
    and and5_inst(and5, A[0], B[2]);
    full_adder fa1(.A(and5), .B(s1), .Cin(c0), .S( P[2]    ), .Cout(c2));
    and and6_inst(and6, A[2], B[1]);
    and and7_inst(and7, A[1], B[2]);
    wire s3,c3;
    full_adder fa2(.A(and7), .B(and6), .Cin(c1), .S(s3    ), .Cout(c3));
    wire c4;
    half_adder ha3(.A(s3), .B(c2), .S(P[3]), .C(c4));
    and and8_inst(and8, A[2], B[2]);
    full_adder fa3(.A(and8), .B(c3), .Cin(c4), .S(P[4]), .Cout(P[5]));
	// Your code goes here.  DO NOT change anything that is already given! Otherwise, you will not be able to pass the tests!


endmodule
