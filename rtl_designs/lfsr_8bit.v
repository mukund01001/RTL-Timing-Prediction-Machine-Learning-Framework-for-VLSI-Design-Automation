// lfsr_8bit.v (Linear Feedback Shift Register)
module lfsr_8bit(
    input clk, rst,
    output reg [7:0] random_out
);
    wire feedback;
    assign feedback = random_out[7] ^ random_out[5] ^ random_out[4] ^ random_out[3];
    
    always @(posedge clk or posedge rst) begin
        if (rst)
            random_out <= 8'b10101010;  // Seed
        else
            random_out <= {random_out[6:0], feedback};
    end
endmodule
