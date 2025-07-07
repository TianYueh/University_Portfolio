//110550085
//Subject:     CO project 2 - Simple Single CPU
//--------------------------------------------------------------------------------
//Version:     1
//--------------------------------------------------------------------------------
//Writer:      
//----------------------------------------------
//Date:        
//----------------------------------------------
//Description: 
//--------------------------------------------------------------------------------
`timescale 1ns/1ps
module Simple_Single_CPU(
        clk_i,
		rst_i
		);
		
//I/O port
input         clk_i;
input         rst_i;

//Internal Signals
wire [32-1:0] pc_in_i, pc_out_o;
wire [32-1:0] adder1_out_o, adder2_out_o;
wire [32-1:0] instruction;
wire RegDst, RegWrite, branch, ALUSrc;
wire [3-1:0] ALUOp;
wire [5-1:0] WriteReg1;
wire [32-1:0] ReadData1, ReadData2;
wire [4-1:0] ALUCtrl;
wire [32-1:0] SignExtend;
wire [32-1:0] ALUIn;
wire [32-1:0] ALUResult;
wire [32-1:0] SignExtend_Shifted;
wire zero;


//Greate components
ProgramCounter PC(
        .clk_i(clk_i),      
	    .rst_i (rst_i),     
	    .pc_in_i(pc_in_i) ,   
	    .pc_out_o(pc_out_o) 
	    );
	
Adder Adder1(
        .src1_i(pc_out_o),     
	    .src2_i(32'd4),     
	    .sum_o(adder1_out_o)    
	    );
	
Instr_Memory IM(
        .pc_addr_i(pc_out_o),  
	    .instr_o(instruction)    
	    );

MUX_2to1 #(.size(5)) Mux_Write_Reg(
        .data0_i(instruction[20:16]),
        .data1_i(instruction[15:11]),
        .select_i(RegDst),
        .data_o(WriteReg1)
        );	
		
Reg_File RF(
        .clk_i(clk_i),      
	    .rst_i(rst_i) ,     
        .RSaddr_i(instruction[25:21]) ,  
        .RTaddr_i(instruction[20:16]) ,  
        .RDaddr_i(WriteReg1) ,  
        .RDdata_i(ALUResult)  , 
        .RegWrite_i (RegWrite),
        .RSdata_o(ReadData1) ,  
        .RTdata_o(ReadData2)   
        );
	
Decoder Decoder(
        .instr_op_i(instruction[31:26]), 
	    .RegWrite_o(RegWrite), 
	    .ALU_op_o(ALUOp),   
	    .ALUSrc_o(ALUSrc),   
	    .RegDst_o(RegDst),   
		.Branch_o(branch)   
	    );

ALU_Ctrl AC(
        .funct_i(instruction[6-1:0]),   
        .ALUOp_i(ALUOp),   
        .ALUCtrl_o(ALUCtrl) 
        );
	
Sign_Extend SE(
        .data_i(instruction[16-1:0]),
        .data_o(SignExtend)
        );

MUX_2to1 #(.size(32)) Mux_ALUSrc(
        .data0_i(ReadData2),
        .data1_i(SignExtend),
        .select_i(ALUSrc),
        .data_o(ALUIn)
        );	
		
ALU ALU(
        .src1_i(ReadData1),
	    .src2_i(ALUIn),
	    .ctrl_i(ALUCtrl),
	    .result_o(ALUResult),
		.zero_o(zero)
	    );
		
Adder Adder2(
        .src1_i(SignExtend_Shifted),     
	    .src2_i(adder1_out_o),     
	    .sum_o(adder2_out_o)      
	    );
		
Shift_Left_Two_32 Shifter(
        .data_i(SignExtend),
        .data_o(SignExtend_Shifted)
        ); 		
		
MUX_2to1 #(.size(32)) Mux_PC_Source(
        .data0_i(adder1_out_o),
        .data1_i(adder2_out_o),
        .select_i(branch&zero),
        .data_o(pc_in_i)
        );	

endmodule



  


