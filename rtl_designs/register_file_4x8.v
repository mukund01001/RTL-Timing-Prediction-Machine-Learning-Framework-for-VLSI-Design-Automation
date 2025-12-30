// register_file_4x8.v (4 registers, 8-bit each)
module register_file_4x8(
    input clk, rst, wr_en,
    input [1:0] wr_addr, rd_addr,
    input [7:0] wr_data,
    output [7:0] rd_data
);
    reg [7:0] regs [0:3];
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            regs[0] <= 8'd0;
            regs[1] <= 8'd0;
            regs[2] <= 8'd0;
            regs[3] <= 8'd0;
        end else if (wr_en) begin
            regs[wr_addr] <= wr_data;
        end
    end
    
    assign rd_data = regs[rd_addr];
endmodule
