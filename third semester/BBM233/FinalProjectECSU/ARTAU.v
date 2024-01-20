`timescale 1us / 1ps

module ARTAU(
    input CLK,
    input RST,
    input scan_for_target,
    input radar_echo,
    input [31:0] jet_speed,
    input [31:0] max_safe_distance,
    output reg radar_pulse_trigger,
    output reg [31:0] distance_to_target,
    output reg threat_detected,
    output reg [1:0] ARTAU_state
);

  // Define states
  parameter IDLE = 2'b00;
  parameter EMIT = 2'b01;
  parameter LISTEN = 2'b10;
  parameter ASSESS = 2'b11;
    initial begin
        ARTAU_state = IDLE;
        next_state = IDLE;
        radar_pulse_trigger = 1'b0;
        distance_to_target = 32'b0;
        threat_detected = 1'b0;
    end
    integer pulse_count = 0;
    reg [1:0] next_state;
    reg clk = 0;
    always #1 clk = ~clk;
    reg change = 0;
    always @(posedge CLK or posedge RST) // always block to update state
    begin
        if (RST)begin
            next_state = IDLE;
            change = 1;
        end
        if (CLK)begin
            ARTAU_state = next_state;
            if (change)begin
                change = 0;
                radar_pulse_trigger <= 1'b0;
                distance_to_target <= 32'b0;
                threat_detected <= 1'b0;
                pulse_count <= 0;
                ARTAU_state <= IDLE;
                next_state <= IDLE;
            end
        end
    end
    integer radr_time;
    integer radr_time_diff;
    integer echo_time;
    integer echo_time_diff;
    integer status_time;
    integer status_time_diff;
    reg [31:0] distance_to_target2;
    real deneme3;
    real deneme4;
    integer dis_dif ;
    integer haha;
    reg isChanged = 0;
    always @(clk) // always block to COMPUTE next state
    begin
        haha = $time;
        haha = haha-50;
        radr_time_diff = $time - radr_time;
        status_time_diff = $time - status_time;
        echo_time_diff = $time - echo_time;
        if (status_time_diff == 3000 & isChanged)begin
            threat_detected = 1'b0;
            distance_to_target = 32'b0;
        end
        if (radar_pulse_trigger & (radr_time_diff == 300))begin
            radar_pulse_trigger = 1'b0;
        end
        if (scan_for_target & ARTAU_state==next_state)begin
            radar_pulse_trigger = 1'b1;
        end
        case (ARTAU_state)
            IDLE:
                begin
                    if (scan_for_target & ARTAU_state==next_state)begin
                        radar_pulse_trigger = 1'b1;
                        //set it to current time
                        pulse_count = pulse_count + 1;
                        radr_time = $time;
                    end
                    if (scan_for_target & ARTAU_state==next_state)begin
                        next_state = EMIT;     
                    end                   
                end
            EMIT:
                begin
                    if (radr_time_diff == 300 )begin
                        //set it to current time
                        echo_time = $time;
                    end
                    if (radr_time_diff == 300 )begin
                        next_state = LISTEN;
                    end
                end
            LISTEN:
                begin

                    if (radar_echo == 1 & pulse_count == 1   & ARTAU_state==next_state)begin
                        radar_pulse_trigger = 1'b1;
                        if (pulse_count > 0)begin
                             distance_to_target = 150 * echo_time_diff;
                             isChanged = 0;
                             dis_dif = $time;
                        end
                        //set it to current time
                        pulse_count <= pulse_count + 1;
                        radr_time = $time;
                    end
                    else if (radar_echo == 1 & pulse_count == 2  & ARTAU_state==next_state)begin
                        distance_to_target2 = 150 * echo_time_diff;
                        deneme4 = ($time-dis_dif)/1000000;
                        deneme4 = deneme4*jet_speed;
                        deneme3 = (distance_to_target2+deneme4-distance_to_target);
                        isChanged = 0;
                        if (distance_to_target2 < max_safe_distance & deneme3 < 0  )begin
                            threat_detected = 1'b1;
                        end
                        pulse_count <= 0;
                        status_time = $time;
                        distance_to_target = distance_to_target2;
                        dis_dif = $time;
                    end
                    else if (echo_time_diff == 2000 & ARTAU_state==next_state)begin
                        pulse_count <= 0;
                        distance_to_target = 32'b0;
                        threat_detected = 1'b0;
                        isChanged = 0;
                        radar_pulse_trigger = 1'b0;
                    end

                    if (radar_echo == 1 & pulse_count == 1  & ARTAU_state==next_state)begin
                        next_state = EMIT;
                    end
                    else if (radar_echo == 1 & pulse_count == 2  & ARTAU_state==next_state)begin
                        next_state = ASSESS;
                    end
                    else if (echo_time_diff == 2000 & ARTAU_state==next_state)begin
                        next_state = IDLE;
                    end
                end

            ASSESS:
                begin

                     if (scan_for_target  & ARTAU_state==next_state)begin
                        radar_pulse_trigger = 1'b1;
                        pulse_count = pulse_count + 1;
                        radr_time = $time;
                        isChanged = 1;
                    end
                    else if (status_time_diff == 3000 & scan_for_target == 0 )begin
                        threat_detected = 1'b0;
                        distance_to_target = 32'b0;
                        pulse_count = 0;
                        radar_pulse_trigger = 1'b0;
                    end
                     if (scan_for_target  & ARTAU_state==next_state)begin
                        next_state = EMIT;     
                    end       
                    else if (status_time_diff == 3000 & scan_for_target == 0 & ARTAU_state==next_state)begin
                        next_state = IDLE;
                    end


                end

        endcase
        if (haha%100 == 0)begin
            ARTAU_state = next_state;
        end
    end

endmodule