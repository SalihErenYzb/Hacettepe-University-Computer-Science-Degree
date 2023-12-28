`timescale 1ns/10ps

module full_adder_tb;
    // Your code goes here.  DO NOT change anything that is already given! Otherwise, you will not be able to pass the tests!
    reg A, B, Cin;
    wire S, Cout;
    full_adder tmp(.A(A), .B(B), .Cin(Cin), .S(S), .Cout(Cout));
    initial begin
        A = 1'b0; B= 1'b0; Cin = 1'b0;
        #10 A = 1'b0; B= 1'b0; Cin = 1'b1;
        #10 A = 1'b0; B= 1'b1; Cin = 1'b0;
        #10 A = 1'b0; B= 1'b1; Cin = 1'b1;
        #10 A = 1'b1; B= 1'b0; Cin = 1'b0;
        #10 A = 1'b1; B= 1'b0; Cin = 1'b1;
        #10 A = 1'b1; B= 1'b1; Cin = 1'b0;
        #10 A = 1'b1; B= 1'b1; Cin = 1'b1;
        #10 $finish;
    end
endmodule
