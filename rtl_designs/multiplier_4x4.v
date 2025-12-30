// multiplier_4x4.v
module multiplier_4x4(
    input [3:0] a, b,
    output [7:0] product
);
    wire [3:0] p0, p1, p2, p3;
    wire [7:0] sum1, sum2, sum3;
    
    // Partial products
    assign p0 = {4{a[0]}} & b;
    assign p1 = {4{a[1]}} & b;
    assign p2 = {4{a[2]}} & b;
    assign p3 = {4{a[3]}} & b;
    
    // Add partial products
    assign sum1 = {4'b0, p0};
    assign sum2 = sum1 + {3'b0, p1, 1'b0};
    assign sum3 = sum2 + {2'b0, p2, 2'b0};
    assign product = sum3 + {1'b0, p3, 3'b0};
endmodule
