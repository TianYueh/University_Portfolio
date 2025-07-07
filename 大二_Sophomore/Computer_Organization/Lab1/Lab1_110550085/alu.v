`timescale 1ns/1ps
// 110550085
module alu(
    /* input */
    clk,            // system clock
    rst_n,          // negative reset
    src1,           // 32 bits, source 1
    src2,           // 32 bits, source 2
    ALU_control,    // 4 bits, ALU control input
    /* output */
    result,         // 32 bits, result
    zero,           // 1 bit, set to 1 when the output is 0
    cout,           // 1 bit, carry out
    overflow        // 1 bit, overflow
);

/*==================================================================*/
/*                          input & output                          */
/*==================================================================*/

input clk;
input rst_n;
input [31:0] src1;
input [31:0] src2;
input [3:0] ALU_control;

output [32-1:0] result;
output zero;
output cout;
output overflow;

/*==================================================================*/
/*                            reg & wire                            */
/*==================================================================*/

reg [32-1:0] result;
reg zero, cout, overflow;

wire [32-1:0] result1;
wire [32-1:0] carry;
wire of;
wire [1:0] operation1;
wire a_input, b_input;





/*==================================================================*/
/*                              design                              */
/*==================================================================*/
assign operation1=ALU_control[1:0];
assign a_input=ALU_control[3];
assign b_input=ALU_control[2];


always@(posedge clk or negedge rst_n) 
begin
	if(!rst_n) begin
	   result=0;
	   zero=0;
	   cout=0;
	   overflow=0;

	end
	else begin
        result=result1;
        zero=~|result;//nor operation, if all bits are 0, then output would be 1
        overflow=carry[30]^carry[31];
        if(operation1==2'b10)begin
            cout=carry[31];
        end
        else begin
            cout=0;
        end
        
	end
end

// HINT: You may use alu_top as submodule.

alu_top ALU00(.src1(src1[0]), .src2(src2[0]), .less(set), .A_invert(a_input), .B_invert(b_input), .cin(b_input), .operation(operation1), .result(result1[0]), .cout(carry[0]));
alu_top ALU01(.src1(src1[1]), .src2(src2[1]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[0]), .operation(operation1), .result(result1[1]), .cout(carry[1]));
alu_top ALU02(.src1(src1[2]), .src2(src2[2]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[1]), .operation(operation1), .result(result1[2]), .cout(carry[2]));
alu_top ALU03(.src1(src1[3]), .src2(src2[3]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[2]), .operation(operation1), .result(result1[3]), .cout(carry[3]));
alu_top ALU04(.src1(src1[4]), .src2(src2[4]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[3]), .operation(operation1), .result(result1[4]), .cout(carry[4]));
alu_top ALU05(.src1(src1[5]), .src2(src2[5]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[4]), .operation(operation1), .result(result1[5]), .cout(carry[5]));
alu_top ALU06(.src1(src1[6]), .src2(src2[6]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[5]), .operation(operation1), .result(result1[6]), .cout(carry[6]));
alu_top ALU07(.src1(src1[7]), .src2(src2[7]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[6]), .operation(operation1), .result(result1[7]), .cout(carry[7]));
alu_top ALU08(.src1(src1[8]), .src2(src2[8]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[7]), .operation(operation1), .result(result1[8]), .cout(carry[8]));
alu_top ALU09(.src1(src1[9]), .src2(src2[9]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[8]), .operation(operation1), .result(result1[9]), .cout(carry[9]));
alu_top ALU10(.src1(src1[10]), .src2(src2[10]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[9]), .operation(operation1), .result(result1[10]), .cout(carry[10]));
alu_top ALU11(.src1(src1[11]), .src2(src2[11]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[10]), .operation(operation1), .result(result1[11]), .cout(carry[11]));
alu_top ALU12(.src1(src1[12]), .src2(src2[12]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[11]), .operation(operation1), .result(result1[12]), .cout(carry[12]));
alu_top ALU13(.src1(src1[13]), .src2(src2[13]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[12]), .operation(operation1), .result(result1[13]), .cout(carry[13]));
alu_top ALU14(.src1(src1[14]), .src2(src2[14]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[13]), .operation(operation1), .result(result1[14]), .cout(carry[14]));
alu_top ALU15(.src1(src1[15]), .src2(src2[15]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[14]), .operation(operation1), .result(result1[15]), .cout(carry[15]));
alu_top ALU16(.src1(src1[16]), .src2(src2[16]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[15]), .operation(operation1), .result(result1[16]), .cout(carry[16]));
alu_top ALU17(.src1(src1[17]), .src2(src2[17]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[16]), .operation(operation1), .result(result1[17]), .cout(carry[17]));
alu_top ALU18(.src1(src1[18]), .src2(src2[18]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[17]), .operation(operation1), .result(result1[18]), .cout(carry[18]));
alu_top ALU19(.src1(src1[19]), .src2(src2[19]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[18]), .operation(operation1), .result(result1[19]), .cout(carry[19]));
alu_top ALU20(.src1(src1[20]), .src2(src2[20]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[19]), .operation(operation1), .result(result1[20]), .cout(carry[20]));
alu_top ALU21(.src1(src1[21]), .src2(src2[21]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[20]), .operation(operation1), .result(result1[21]), .cout(carry[21]));
alu_top ALU22(.src1(src1[22]), .src2(src2[22]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[21]), .operation(operation1), .result(result1[22]), .cout(carry[22]));
alu_top ALU23(.src1(src1[23]), .src2(src2[23]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[22]), .operation(operation1), .result(result1[23]), .cout(carry[23]));
alu_top ALU24(.src1(src1[24]), .src2(src2[24]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[23]), .operation(operation1), .result(result1[24]), .cout(carry[24]));
alu_top ALU25(.src1(src1[25]), .src2(src2[25]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[24]), .operation(operation1), .result(result1[25]), .cout(carry[25]));
alu_top ALU26(.src1(src1[26]), .src2(src2[26]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[25]), .operation(operation1), .result(result1[26]), .cout(carry[26]));
alu_top ALU27(.src1(src1[27]), .src2(src2[27]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[26]), .operation(operation1), .result(result1[27]), .cout(carry[27]));
alu_top ALU28(.src1(src1[28]), .src2(src2[28]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[27]), .operation(operation1), .result(result1[28]), .cout(carry[28]));
alu_top ALU29(.src1(src1[29]), .src2(src2[29]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[28]), .operation(operation1), .result(result1[29]), .cout(carry[29]));
alu_top ALU30(.src1(src1[30]), .src2(src2[30]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[29]), .operation(operation1), .result(result1[30]), .cout(carry[30]));
alu_top_overdetect ALU31(.src1(src1[31]), .src2(src2[31]), .less(0), .A_invert(a_input), .B_invert(b_input), .cin(carry[30]), .operation(operation1), .result(result1[31]), .cout(carry[31]), .set(set));


endmodule
