//110550085
//Subject:     CO project 2 - Decoder
//--------------------------------------------------------------------------------
//Version:     1
//--------------------------------------------------------------------------------
//Writer:      Luke
//----------------------------------------------
//Date:        
//----------------------------------------------
//Description: 
//--------------------------------------------------------------------------------
`timescale 1ns/1ps
module Decoder(
    instr_op_i,
	RegWrite_o,
	ALU_op_o,
	ALUSrc_o,
	RegDst_o,
	Branch_o
	);
     
//I/O ports
input  [6-1:0] instr_op_i;

output         RegWrite_o;
output [3-1:0] ALU_op_o;
output         ALUSrc_o;
output         RegDst_o;
output         Branch_o;
 
//Internal Signals
reg    [3-1:0] ALU_op_o;
reg            ALUSrc_o;
reg            RegWrite_o;
reg            RegDst_o;
reg            Branch_o;

//Parameter


//Main function
always@(*)begin
    //R-format
    if(instr_op_i==6'b000000)begin
        ALU_op_o=3'b010;
        RegWrite_o=1'b1;
        ALUSrc_o=1'b0;
        RegDst_o=1'b1;
        Branch_o=1'b0;
    end
    //addi
    else if(instr_op_i==6'b001000)begin
        ALU_op_o=3'b000;
        RegWrite_o=1'b1;
        ALUSrc_o=1'b1;
        RegDst_o=1'b0;
        Branch_o=1'b0;
    end
    //beq
    else if(instr_op_i==6'b000100)begin
        ALU_op_o=3'b001;
        RegWrite_o=1'b0;
        ALUSrc_o=1'b0;
        RegDst_o=1'b0;
        Branch_o=1'b1;
    end
    //slti
    else if(instr_op_i==6'b001010)begin
        ALU_op_o=3'b011;
        RegWrite_o=1'b1;
        ALUSrc_o=1'b1;
        RegDst_o=1'b0;
        Branch_o=1'b0;
    end
    else begin
        ALU_op_o=3'bxxx;
        RegWrite_o=1'bx;
        ALUSrc_o=1'bx;
        RegDst_o=1'bx;
        Branch_o=1'bx;
    end

end


endmodule





                    
                    