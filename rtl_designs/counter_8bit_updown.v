// counter_8bit_updown.v
module counter_8bit_updown(
    input clk, rst, up_down,  // up_down: 1=up, 0=down
    output reg [7:0] count
);
    always @(posedge clk or posedge rst) begin
        if (rst)
            count <= 8'b0;
        else if (up_down)
            count <= count + 1;
        else
            count <= count - 1;
    end
endmodule
