module jkff (
    input J,      // Data input
    input K,      // Data input
    input CLK,    // Clock input
    input RESET,  // Asynchronous reset, active high
    output reg Q  // Output
);
    // Your code goes here.  DO NOT change anything that is already given! Otherwise, you will not be able to pass the tests!
    always @(posedge CLK or posedge RESET) begin
        if (RESET) begin
            Q <= 1'b0;
        end else begin
            if (J && K) begin
                Q <= ~Q;
            end else if (J) begin
                Q <= 1'b1;
            end else if (K) begin
                Q <= 1'b0;
            end else begin
                Q <= Q;
            end
        end
    end
endmodule