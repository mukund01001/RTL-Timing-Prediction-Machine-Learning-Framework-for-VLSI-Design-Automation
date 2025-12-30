// fsm_sequence_detector.v (Detects 1011)
module fsm_sequence_detector(
    input clk, rst, in,
    output reg detected
);
    localparam S0 = 3'd0, S1 = 3'd1, S2 = 3'd2, S3 = 3'd3, S4 = 3'd4;
    reg [2:0] state, next_state;
    
    always @(posedge clk or posedge rst) begin
        if (rst)
            state <= S0;
        else
            state <= next_state;
    end
    
    always @(*) begin
        next_state = state;
        detected = 1'b0;
        case(state)
            S0: next_state = in ? S1 : S0;
            S1: next_state = in ? S1 : S2;
            S2: next_state = in ? S3 : S0;
            S3: begin
                next_state = in ? S4 : S2;
                detected = in;
            end
            S4: next_state = in ? S1 : S2;
        endcase
    end
endmodule
