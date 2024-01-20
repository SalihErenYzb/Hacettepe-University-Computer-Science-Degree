`timescale 1us / 1ps

module ECSU(
    input CLK,
    input RST,
    input thunderstorm,
    input [5:0] wind,
    input [1:0] visibility,
    input signed [7:0] temperature,
    output reg severe_weather,
    output reg emergency_landing_alert,
    output reg [1:0] ECSU_state
);

// Your code goes here.
    reg [1:0] nextstate; 
    initial begin
        severe_weather = 0;
        emergency_landing_alert = 0;
        ECSU_state = 2'b00;
        nextstate = 2'b00;
    end
    reg change = 0;
    always @(posedge CLK or posedge RST) 
    begin
        if (RST) 
        begin
            nextstate = 2'b00;
            change = 1;
        end
        if (CLK)  
        begin
            if (change)
            begin
                severe_weather = 0;
                emergency_landing_alert = 0;
                change = 0;
            end
            ECSU_state = nextstate; 
        end
    end
    always @(ECSU_state or wind or visibility or temperature or thunderstorm ) // always block to compute output
    begin
        case(ECSU_state)
            2'b00: begin
                if (thunderstorm == 1 | wind > 15 | temperature > 35 | temperature < -35 | visibility == 2'b11) begin
                    severe_weather = 1;
                end
            end
            2'b01: begin
                if (thunderstorm == 1 | wind > 15 | temperature > 35 | temperature < -35 | visibility == 2'b11) begin
                    severe_weather = 1;
                end
            end
            2'b10: begin
                if (thunderstorm == 0 & wind <= 10 & temperature >= -35 & temperature <= 35 & visibility == 2'b01) begin
                    severe_weather = 0;
                end
                if (temperature < -40 | temperature > 40 | wind > 20) begin
                    emergency_landing_alert = 1;
                end
            end

        endcase
    end 
    always @(ECSU_state or wind or visibility or temperature or thunderstorm ) // always block to compute nextstate
    begin
        case(ECSU_state)
            2'b00: begin
                if (thunderstorm == 1 | wind > 15 | temperature > 35 | temperature < -35 | visibility == 2'b11) begin
                    nextstate = 2'b10;
                end
                if ((wind <= 15 & wind > 10) | visibility == 2'b01) begin
                    nextstate = 2'b01;
                end
            end
            2'b01: begin
                if (thunderstorm == 1 | wind > 15 | temperature > 35 | temperature < -35 | visibility == 2'b11) begin
                    nextstate = 2'b10;
                end
                if (visibility == 2'b00 & wind <= 10) begin
                    nextstate = 2'b00;
                end
            end
            2'b10: begin
                if (thunderstorm == 0 & wind <= 10 & temperature >= -35 & temperature <= 35 & visibility == 2'b01) begin
                     nextstate = 2'b01;
                end
                if (temperature < -40 | temperature > 40 | wind > 20) begin
                    nextstate = 2'b11;
                end
            end
            
        endcase
    end
endmodule