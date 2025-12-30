// mux_4to1.v
module mux_4to1(
    input [3:0] in0, in1, in2, in3,
    input [1:0] sel,
    output [3:0] out
);
    assign out = (sel == 2'b00) ? in0 :
                 (sel == 2'b01) ? in1 :
                 (sel == 2'b10) ? in2 : in3;
endmodule
