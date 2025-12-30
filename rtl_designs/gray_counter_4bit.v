// gray_counter_4bit.v
module gray_counter_4bit(
    input clk, rst,
    output reg [3:0] gray_count
);
    reg [3:0] binary_count;
    
    always @(posedge clk or posedge rst) begin
        if (rst)
            binary_count <= 4'd0;
        else
            binary_count <= binary_count + 1;
    end
    
    // Binary to Gray conversion
    always @(*) begin
        gray_count[3] = binary_count[3];
        gray_count[2] = binary_count[3] ^ binary_count[2];
        gray_count[1] = binary_count[2] ^ binary_count[1];
        gray_count[0] = binary_count[1] ^ binary_count[0];
    end
endmodule
