// adder_4bit_cla.v (Carry-Lookahead Adder)
module adder_4bit_cla(
    input [3:0] a, b,
    input cin,
    output [3:0] sum,
    output cout
);
    wire [3:0] g, p;  // Generate and Propagate
    wire [3:0] c;     // Carries
    
    // Generate and Propagate
    assign g = a & b;
    assign p = a ^ b;
    
    // Carry calculation
    assign c[0] = cin;
    assign c[1] = g[0] | (p[0] & c[0]);
    assign c[2] = g[1] | (p[1] & g[0]) | (p[1] & p[0] & c[0]);
    assign c[3] = g[2] | (p[2] & g[1]) | (p[2] & p[1] & g[0]) | (p[2] & p[1] & p[0] & c[0]);
    assign cout = g[3] | (p[3] & g[2]) | (p[3] & p[2] & g[1]) | (p[3] & p[2] & p[1] & g[0]) | 
                  (p[3] & p[2] & p[1] & p[0] & c[0]);
    
    // Sum calculation
    assign sum = p ^ c;
endmodule
