`timescale 1ns/1ps
// 
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

/*==================================================================*/
/*                              design                              */
/*==================================================================*/

always@(posedge clk or negedge rst_n) 
begin
	if(!rst_n) begin

	end
	else begin

	end
end

/* HINT: You may use alu_top as submodule.
// 32-bit ALU
alu_top ALU00(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU01(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU02(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU03(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU04(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU05(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU06(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU07(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU08(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU09(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU10(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU11(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU12(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU13(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU14(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU15(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU16(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU17(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU18(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU19(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU20(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU21(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU22(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU23(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU24(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU25(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU26(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU27(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU28(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU29(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU30(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
alu_top ALU31(.src1(), .src2(), .less(), .A_invert(), .B_invert(), .cin(), .operation(), .result(), .cout());
*/

endmodule
