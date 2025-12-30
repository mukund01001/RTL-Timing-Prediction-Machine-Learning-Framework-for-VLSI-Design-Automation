// fifo_8x4.v (8 entries, 4-bit wide)
module fifo_8x4(
    input clk, rst, wr_en, rd_en,
    input [3:0] data_in,
    output reg [3:0] data_out,
    output reg full, empty
);
    reg [3:0] mem [0:7];
    reg [2:0] wr_ptr, rd_ptr;
    reg [3:0] count;
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            wr_ptr <= 3'd0;
            rd_ptr <= 3'd0;
            count <= 4'd0;
            full <= 1'b0;
            empty <= 1'b1;
        end else begin
            if (wr_en && !full) begin
                mem[wr_ptr] <= data_in;
                wr_ptr <= wr_ptr + 1;
                count <= count + 1;
            end
            if (rd_en && !empty) begin
                data_out <= mem[rd_ptr];
                rd_ptr <= rd_ptr + 1;
                count <= count - 1;
            end
            full <= (count == 4'd8);
            empty <= (count == 4'd0);
        end
    end
endmodule
