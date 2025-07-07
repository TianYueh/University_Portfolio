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
module Simple_Single_CPU(
        clk_i,
		rst_i
		);
		
//I/O port
input         clk_i;
input         rst_i;

//Internal Signals
wire [32-1:0] pc_in_i, pc_out_o;
wire [32-1:0] pc_plus_four_o;
wire [32-1:0] instr_o;
wire [2-1:0] RegDst;
wire [5-1:0] WriteRegister;
wire [32-1:0] WriteData;
wire [32-1:0] ReadData1, ReadData2;
wire [3-1:0] ALUop;
wire RegWrite;
wire ALUsrc;
wire Branch;
wire [2-1:0] Jump;
wire MemRead;
wire MemWrite;
wire [2-1:0] MemtoReg;
wire [2-1:0] BranchType;
wire [4-1:0] ALUCtrl_o;
wire [32-1:0] Sign_Extended;
wire [32-1:0] ALU_in;
wire [32-1:0] ALU_Result;
wire zero;
wire [32-1:0] DM_ReadData;
wire [32-1:0] Extended_Shifted;
wire [32-1:0] pc_shifted_added;
wire The_Four;
wire [32-1:0] Jump_o;


//Create components
ProgramCounter PC(
        .clk_i(clk_i),      
	    .rst_i (rst_i),     
	    .pc_in_i(pc_in_i) ,   
	    .pc_out_o(pc_out_o) 
	    );
	
Adder Adder1(
        .src1_i(pc_out_o),     
	    .src2_i(32'd4),     
	    .sum_o(pc_plus_four_o)    
	    );
	
Instr_Memory IM(
        .pc_addr_i(pc_out_o),  
	    .instr_o(instr_o)    
	    );

MUX_3to1 #(.size(5)) Mux_Write_Reg(
        .data0_i(instr_o[20:16]),
        .data1_i(instr_o[15:11]),
        .data2_i(5'b11111),
        .select_i(RegDst),
        .data_o(WriteRegister)
        );	
		
Reg_File Registers(
        .clk_i(clk_i),      
	    .rst_i(rst_i) ,     
        .RSaddr_i(instr_o[25:21]) ,  
        .RTaddr_i(instr_o[20:16]) ,  
        .RDaddr_i(WriteRegister) ,  
        .RDdata_i(WriteData)  , 
        .RegWrite_i (RegWrite),
        .RSdata_o(ReadData1) ,  
        .RTdata_o(ReadData2)   
        );
	
Decoder Decoder(
        .instr_op_i(instr_o[31:26]),
        .funct_i(instr_o[5:0]),
	    .RegWrite_o(RegWrite),
	    .ALU_op_o(ALUop),
	    .ALUSrc_o(ALUsrc),
	    .RegDst_o(RegDst),
	    .Branch_o(Branch),
	    .Jump_o(Jump),
	    .MemRead_o(MemRead),
	    .MemWrite_o(MemWrite),
	    .MemtoReg_o(MemtoReg),
	    .BranchType_o(BranchType)
	    );

ALU_Ctrl AC(
        .funct_i(instr_o[5:0]),   
        .ALUOp_i(ALUop),   
        .ALUCtrl_o(ALUCtrl_o) 
        );
	
Sign_Extend SE(
        .data_i(instr_o[15:0]),
        .data_o(Sign_Extended)
        );

MUX_2to1 #(.size(32)) Mux_ALUSrc(
        .data0_i(ReadData2),
        .data1_i(Sign_Extended),
        .select_i(ALUsrc),
        .data_o(ALU_in)
        );	
		
ALU ALU(
        .src1_i(ReadData1),
	    .src2_i(ALU_in),
	    .ctrl_i(ALUCtrl_o),
	    .result_o(ALU_Result),
		.zero_o(zero)
	    );
	
Data_Memory Data_Memory(
	.clk_i(clk_i),
	.addr_i(ALU_Result),
	.data_i(ReadData2),
	.MemRead_i(MemRead),
	.MemWrite_i(MemWrite),
	.data_o(DM_ReadData)
	);
	
Adder Adder2(
        .src1_i(pc_plus_four_o),     
	    .src2_i(Extended_Shifted),     
	    .sum_o(pc_shifted_added)      
	    );
		
Shift_Left_Two_32 Shifter(
        .data_i(Sign_Extended),
        .data_o(Extended_Shifted)
        ); 		
		
MUX_3to1 #(.size(32)) Mux_PC_Source(
        .data0_i({pc_plus_four_o[31:28], instr_o[25:0], 2'b00}),
        .data1_i(Jump_o),
        .data2_i(ReadData1),
        .select_i(Jump),
        .data_o(pc_in_i)
        );	
       
MUX_2to1 #(.size(32)) Mux_Jump(
        .data0_i(pc_plus_four_o),
        .data1_i(pc_shifted_added),
        .select_i(Branch&The_Four),
        .data_o(Jump_o)
        );
        
MUX_4to1 #(.size(1)) Mux_the_four(
        .data0_i(zero),
        .data1_i(~(ALU_Result[31]|zero)),
        .data2_i(~ALU_Result[31]),
        .data3_i(~zero),
        .select_i(BranchType),
        .data_o(The_Four)
        );
        
MUX_4to1 #(.size(32)) Mux_the_right_four(
        .data0_i(ALU_Result),
        .data1_i(DM_ReadData),
        .data2_i(Sign_Extended),
        .data3_i(pc_plus_four_o),
        .select_i(MemtoReg),
        .data_o(WriteData)
        );

endmodule
	


 


