`timescale 1ns / 1ps

module machine_d_tb;
    // Your code goes here.  DO NOT change anything that is already given! Otherwise, you will not be able to pass the tests!
    reg x;
    reg CLK;
    reg RESET;

    wire F;
    wire [2:0] S;
    machine_d uut (
        .x(x),
        .CLK(CLK),
        .RESET(RESET),
        .F(F),
        .S(S),
    );
    initial begin
        CLK = 0;
        RESET = 1;
        x = 0;
        #10 RESET = 0;
        repeat (8) begin
            #5 x = ~x;
            for (j = 0; j < 2; j = j + 1) begin
                #5 CLK = ~CLK;
                for (i = 0; i < 2; i = i + 1) begin
                    #5 RESET = ~RESET;
                end
            end
        end

        #10 $finish;
    end
endmodule