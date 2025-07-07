`timescale 1ns / 1ps
// 110550085
module Pipelined_CPU(
    clk_i,
    rst_i
);
    
/*==================================================================*/
/*                          input & output                          */
/*==================================================================*/

input clk_i;
input rst_i;

/*==================================================================*/
/*                            reg & wire                            */
/*==================================================================*/

/**** IF stage ****/

wire [32-1:0] pc_in_i, pc_out_o;
wire [32-1:0] instr;
wire [32-1:0] pc_plus_four;

/**** ID stage ****/

wire [32-1:0] pc_plus_four_id, instr_id;
wire [32-1:0] WriteData_id;
wire RegWrite_id_i;
wire [32-1:0] ReadData1, ReadData2;

wire RegWrite_id_o;
wire [3-1:0] ALUop;
wire ALUsrc;
wire RegDst;
wire Branch;
wire MemRead;
wire MemWrite;
wire MemtoReg;

wire [32-1:0] Sign_Extended;

/**** EX stage ****/

wire [32-1:0] pc_plus_four_ex;

wire [32-1:0] instr_ex;
wire [32-1:0] ReadData1_ex, ReadData2_ex;

wire RegWrite_ex;
wire [3-1:0] ALUop_ex;
wire ALUsrc_ex;
wire RegDst_ex;
wire Branch_ex;
wire MemRead_ex;
wire MemWrite_ex;
wire MemtoReg_ex;

wire [32-1:0] Sign_Extended_ex;

wire [32-1:0] Sign_Extended_Shifted;
wire [4-1:0] ALUCtrl;
wire [32-1:0] Mux1_o;
wire [5-1:0] Mux2_o;
wire [32-1:0] ALUResult;
wire zero;
wire [32-1:0] Add_pc_branch_result;


/**** MEM stage ****/

wire RegWrite_mem;
wire MemtoReg_mem;
wire Branch_mem;
wire MemRead_mem;
wire MemWrite_mem;
wire [32-1:0] Add_pc_branch_result_mem;
wire zero_mem;
wire [32-1:0] ALUResult_mem;
wire [32-1:0] ReadData2_mem;
wire [5-1:0] Mux2_o_mem;

wire [32-1:0] DM_Result;


/**** WB stage ****/

wire [32-1:0] DM_Result_wb;
wire MemtoReg_wb;
wire [32-1:0] ALUResult_wb;
wire [5-1:0] Mux2_o_wb;



/*==================================================================*/
/*                              design                              */
/*==================================================================*/

//Instantiate the components in IF stage

MUX_2to1 #(.size(32)) Mux0( // Modify N, which is the total length of input/output
    .data0_i(pc_plus_four),
    .data1_i(Add_pc_branch_result_mem),
    .select_i(Branch_mem & zero_mem),
    .data_o(pc_in_i)
);

ProgramCounter PC(
    .clk_i(clk_i),      
	.rst_i (rst_i),     
	.pc_in_i(pc_in_i) ,   
    .pc_out_o(pc_out_o) 
);

Instruction_Memory IM(
    .addr_i(pc_out_o),
    .instr_o(instr)
);
			
Adder Add_pc(
    .src1_i(pc_out_o),
    .src2_i(32'd4),
    .sum_o(pc_plus_four)
);
		
Pipe_Reg #(.size(32+32)) IF_ID( // Modify N, which is the total length of input/output
    .clk_i(clk_i),
    .rst_i(rst_i),
    .data_i({pc_plus_four, instr}),
    .data_o({pc_plus_four_id, instr_id})
);


//Instantiate the components in ID stage

Reg_File RF(
    .clk_i(clk_i),      
	.rst_i(rst_i) ,     
    .RSaddr_i(instr_id[25:21]) ,  
    .RTaddr_i(instr_id[20:16]) ,  
    .RDaddr_i(Mux2_o_wb) ,  
    .RDdata_i(WriteData_id)  , 
    .RegWrite_i(RegWrite_id_i),
    .RSdata_o(ReadData1) ,  
    .RTdata_o(ReadData2)   
);

Decoder Control(
    .instr_op_i(instr_id[31:26]),
    .RegWrite_o(RegWrite_id_o),
	.ALU_op_o(ALUop),
    .ALUSrc_o(ALUsrc),
	.RegDst_o(RegDst),
	.Branch_o(Branch),
	.MemRead_o(MemRead),
	.MemWrite_o(MemWrite),
	.MemtoReg_o(MemtoReg)
);

Sign_Extend SE(
    .data_i(instr_id[15:0]),
    .data_o(Sign_Extended)
);

Pipe_Reg #(.size(32+32+32+32+1+3+1+1+1+1+1+1+32)) ID_EX( // Modify N, which is the total length of input/output
    .clk_i(clk_i),
    .rst_i(rst_i),
    .data_i({pc_plus_four_id, instr_id,  ReadData1, ReadData2, RegWrite_id_o, ALUop, ALUsrc, RegDst, Branch, MemRead, MemWrite, MemtoReg, Sign_Extended}),
    .data_o({pc_plus_four_ex, instr_ex,  ReadData1_ex, ReadData2_ex, RegWrite_ex, ALUop_ex, ALUsrc_ex, RegDst_ex, Branch_ex, MemRead_ex, MemWrite_ex, MemtoReg_ex, Sign_Extended_ex})
);


//Instantiate the components in EX stage

Shift_Left_Two_32 Shifter(
    .data_i(Sign_Extended_ex),
    .data_o(Sign_Extended_Shifted)
);

ALU ALU(
    .src1_i(ReadData1_ex),
    .src2_i(Mux1_o),
    .ctrl_i(ALUCtrl),
    .result_o(ALUResult),
    .zero_o(zero)
);
		
ALU_Ctrl ALU_Control(
    .funct_i(instr_ex[5:0]),
    .ALUOp_i(ALUop_ex),
    .ALUCtrl_o(ALUCtrl)
);

MUX_2to1 #(.size(32)) Mux1( // Modify N, which is the total length of input/output
    .data0_i(ReadData2_ex),
    .data1_i(Sign_Extended_ex),
    .select_i(ALUsrc_ex),
    .data_o(Mux1_o)
);
		
MUX_2to1 #(.size(5)) Mux2( // Modify N, which is the total length of input/output
    .data0_i(instr_ex[20:16]),
    .data1_i(instr_ex[15:11]),
    .select_i(RegDst_ex),
    .data_o(Mux2_o)
);

Adder Add_pc_branch(
    .src1_i(pc_plus_four_ex),
    .src2_i(Sign_Extended_Shifted),
    .sum_o(Add_pc_branch_result)
);

Pipe_Reg #(.size(1+1+1+1+1+32+1+32+32+5)) EX_MEM( // Modify N, which is the total length of input/output
    .clk_i(clk_i),
    .rst_i(rst_i),
    .data_i({RegWrite_ex, MemtoReg_ex, Branch_ex, MemRead_ex, MemWrite_ex, Add_pc_branch_result, zero, ALUResult, ReadData2_ex, Mux2_o}),
    .data_o({RegWrite_mem, MemtoReg_mem, Branch_mem, MemRead_mem, MemWrite_mem, Add_pc_branch_result_mem, zero_mem, ALUResult_mem, ReadData2_mem, Mux2_o_mem})
);


//Instantiate the components in MEM stage

Data_Memory DM(
    .clk_i(clk_i),
    .addr_i(ALUResult_mem),
    .data_i(ReadData2_mem),
    .MemRead_i(MemRead_mem),
    .MemWrite_i(MemWrite_mem),
    .data_o(DM_Result)
);

Pipe_Reg #(.size(32+1+1+32+5)) MEM_WB( // Modify N, which is the total length of input/output
    .clk_i(clk_i),
    .rst_i(rst_i),
    .data_i({DM_Result, RegWrite_mem, MemtoReg_mem, ALUResult_mem, Mux2_o_mem}),
    .data_o({DM_Result_wb, RegWrite_id_i, MemtoReg_wb, ALUResult_wb, Mux2_o_wb})
);


//Instantiate the components in WB stage

MUX_2to1 #(.size(32)) Mux3( // Modify N, which is the total length of input/output
    .data0_i(ALUResult_wb),
    .data1_i(DM_Result_wb),
    .select_i(MemtoReg_wb),
    .data_o(WriteData_id)
);


endmodule

