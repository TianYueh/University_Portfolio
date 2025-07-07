//110550085
//Subject:     CO project 2 - Decoder
//--------------------------------------------------------------------------------
//Version:     1
//--------------------------------------------------------------------------------
//Writer:      Luke
//----------------------------------------------
//Date:        2010/8/16
//----------------------------------------------
//Description: 
//--------------------------------------------------------------------------------

module Decoder(
    instr_op_i,
	RegWrite_o,
	ALU_op_o,
	ALUSrc_o,
	RegDst_o,
	Branch_o,
	MemRead_o,
	MemWrite_o,
	MemtoReg_o,
	BranchType_o
	);
     
//I/O ports
input  [6-1:0] instr_op_i;
output         RegWrite_o;
output [3-1:0] ALU_op_o;
output         ALUSrc_o;
output         RegDst_o;
output         Branch_o;
output         MemRead_o;
output         MemWrite_o;
output         MemtoReg_o;

output [2-1:0] BranchType_o;

 
//Internal Signals
reg    [3-1:0] ALU_op_o;
reg            ALUSrc_o;
reg            RegWrite_o;
reg            RegDst_o;
reg            Branch_o;
reg            MemRead_o;
reg            MemWrite_o;
reg            MemtoReg_o;

reg    [2-1:0] BranchType_o;
//Parameter


//Main function

always@(*)begin
    //R-format
    if(instr_op_i==6'b000000)begin
        ALU_op_o=3'b010;
        RegDst_o=1'b1;
        ALUSrc_o=1'b0;
        RegWrite_o=1'b1;
        Branch_o=1'b0;
        MemRead_o=1'b0;
        MemWrite_o=1'b0;
        MemtoReg_o=1'b0;
    end
    //addi
    else if(instr_op_i==6'b001000)begin
        ALU_op_o=3'b000;
        RegDst_o=1'b0;
        ALUSrc_o=1'b1;
        RegWrite_o=1'b1;
        Branch_o=1'b0;
        MemRead_o=1'b0;
        MemWrite_o=1'b0;
        MemtoReg_o=1'b0;
    end
    //slti
    else if(instr_op_i==6'b001010)begin
        ALU_op_o=3'b011;
        RegDst_o=1'b0;
        ALUSrc_o=1'b1;
        RegWrite_o=1'b1;
        Branch_o=1'b0;
        MemRead_o=1'b0;
        MemWrite_o=1'b0;
        MemtoReg_o=1'b0;
    end
    //lw
    else if(instr_op_i==6'b100011)begin
        ALU_op_o=3'b000;
        RegDst_o=1'b0;
        ALUSrc_o=1'b1;
        RegWrite_o=1'b1;
        Branch_o=1'b0;
        MemRead_o=1'b1;
        MemWrite_o=1'b0;
        MemtoReg_o=1'b1;
    end
    //sw
    else if(instr_op_i==6'b101011)begin
        ALU_op_o=3'b000;
        RegDst_o=1'b0;
        ALUSrc_o=1'b1;
        RegWrite_o=1'b0;
        Branch_o=1'b0;
        MemRead_o=1'b0;
        MemWrite_o=1'b1;
        MemtoReg_o=1'b0;
    end
    //beq
    else if(instr_op_i==6'b000100)begin
        ALU_op_o=3'b001;
        RegDst_o=1'b0;
        ALUSrc_o=1'b0;
        RegWrite_o=1'b0;
        Branch_o=1'b1;
        MemRead_o=1'b0;
        MemWrite_o=1'b0;
        MemtoReg_o=1'b0;
    end
    
    //Lab5
    //bne, bge, or bgt
    else if(instr_op_i==6'b000101||instr_op_i==6'b000001||instr_op_i==6'b000111)begin
        ALU_op_o=3'b001;
        RegDst_o=1'b0;
        ALUSrc_o=1'b0;
        RegWrite_o=1'b0;
        Branch_o=1'b1;
        MemRead_o=1'b0;
        MemWrite_o=1'b0;
        MemtoReg_o=1'b0;
    end
    
    else begin
        ALU_op_o=3'bxxx;
        RegDst_o=1'bx;
        ALUSrc_o=1'bx;
        RegWrite_o=1'bx;
        Branch_o=1'bx;
        MemRead_o=1'bx;
        MemWrite_o=1'bx;
        MemtoReg_o=1'bx;
    end
    
    if(instr_op_i==6'b000101)begin
        BranchType_o=2'b01;
    end
    else if(instr_op_i==6'b000001)begin
        BranchType_o=2'b10;
    end
    else if(instr_op_i==6'b000111)begin
        BranchType_o=2'b11;
    end
    else begin
        BranchType_o=2'b00;
    end
    
end

endmodule






                    
                    