module machine_jk(
    input wire x,
    input wire CLK,
    input wire RESET,
    output wire F,
    output wire [2:0] S
);
    // Your code goes here.  DO NOT change anything that is already given! Otherwise, you will not be able to pass the tests!
    wire xb;
    wire xab;

    and g1(xb,S[1],~x);
    or DA1(xab,S[2],xb);
    jkff DAJK(xab,~xab,CLK,RESET,S[2]);

    wire bx;
    wire xa;
    wire bxr;
    wire araa;
    wire bxxabxr;

    and g2(bx,S[1],x);
    and g3(xa,S[2],~x);
    and g4(bxr,~S[1],~x);
    or g5(araa,bx,xa);
    or g6(bxxabxr,araa,bxr);
    jkff DBJK(bxxabxr,~bxxabxr,CLK,RESET,S[1]);

    wire cxr;
    wire rcx;
    wire rcxr;
    and g7(cxr,S[0],~x);
    and g8(rcx,~S[0],x);
    or C(rcxr,rcx,cxr);
    jkff DCJK(rcxr,~rcxr,CLK,RESET,S[0]);

    and g9(F,S[2],S[1],~S[0]);

endmodule