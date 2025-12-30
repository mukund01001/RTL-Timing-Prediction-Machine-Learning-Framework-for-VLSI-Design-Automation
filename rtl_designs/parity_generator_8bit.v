// parity_generator_8bit.v
module parity_generator_8bit(
    input [7:0] data,
    output even_parity,
    output odd_parity
);
    assign even_parity = ^data;        // XOR of all bits
    assign odd_parity = ~(^data);
endmodule
