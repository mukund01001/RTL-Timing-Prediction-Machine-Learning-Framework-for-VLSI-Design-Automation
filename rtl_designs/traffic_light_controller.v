// traffic_light_controller.v
module traffic_light_controller(
    input clk, rst,
    output reg [2:0] light  // {R, Y, G}
);
    localparam RED = 3'b100, YELLOW = 3'b010, GREEN = 3'b001;
    reg [1:0] state;
    reg [3:0] timer;
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= 2'b00;
            timer <= 4'd0;
            light <= RED;
        end else begin
            timer <= timer + 1;
            case(state)
                2'b00: begin  // RED
                    light <= RED;
                    if (timer == 4'd10) begin
                        state <= 2'b01;
                        timer <= 4'd0;
                    end
                end
                2'b01: begin  // GREEN
                    light <= GREEN;
                    if (timer == 4'd8) begin
                        state <= 2'b10;
                        timer <= 4'd0;
                    end
                end
                2'b10: begin  // YELLOW
                    light <= YELLOW;
                    if (timer == 4'd3) begin
                        state <= 2'b00;
                        timer <= 4'd0;
                    end
                end
            endcase
        end
    end
endmodule
