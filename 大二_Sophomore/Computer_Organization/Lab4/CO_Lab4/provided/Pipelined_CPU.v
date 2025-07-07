`timescale 1ns / 1ps
// TA
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



/**** ID stage ****/



/**** EX stage ****/



/**** MEM stage ****/



/**** WB stage ****/




/*==================================================================*/
/*                              design                              */
/*==================================================================*/

//Instantiate the components in IF stage

MUX_2to1 #(.size(N)) Mux0( // Modify N, which is the total length of input/output

);

ProgramCounter PC(
    
);

Instruction_Memory IM(
    
);
			
Adder Add_pc(
    
);
		
Pipe_Reg #(.size(N)) IF_ID( // Modify N, which is the total length of input/output

);


//Instantiate the components in ID stage

Reg_File RF(
    
);

Decoder Control(
    
);

Sign_Extend SE(
    
);

Pipe_Reg #(.size(N)) ID_EX( // Modify N, which is the total length of input/output

);


//Instantiate the components in EX stage

Shift_Left_Two_32 Shifter(

);

ALU ALU(

);
		
ALU_Ctrl ALU_Control(
    
);

MUX_2to1 #(.size(N)) Mux1( // Modify N, which is the total length of input/output

);
		
MUX_2to1 #(.size(N)) Mux2( // Modify N, which is the total length of input/output

);

Adder Add_pc_branch(
    
);

Pipe_Reg #(.size(N)) EX_MEM( // Modify N, which is the total length of input/output
    
);


//Instantiate the components in MEM stage

Data_Memory DM(

);

Pipe_Reg #(.size(N)) MEM_WB( // Modify N, which is the total length of input/output
    
);


//Instantiate the components in WB stage

MUX_2to1 #(.size(N)) Mux3( // Modify N, which is the total length of input/output

);


endmodule