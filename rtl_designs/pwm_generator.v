// pwm_generator.v
module pwm_generator(
    input clk, rst,
    input [7:0] duty_cycle,  // 0-255
    output reg pwm_out
);
    reg [7:0] counter;
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            counter <= 8'd0;
            pwm_out <= 1'b0;
        end else begin
            counter <= counter + 1;
            pwm_out <= (counter < duty_cycle);
        end
    end
endmodule
