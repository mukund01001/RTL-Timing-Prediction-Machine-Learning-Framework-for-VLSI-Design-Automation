// counter_4bit_up.v
module counter_4bit_up(
    input clk, rst,
    output reg [3:0] count
);
    always @(posedge clk or posedge rst) begin
        if (rst)
            count <= 4'b0;
        else
            count <= count + 1;
    end
endmodule
