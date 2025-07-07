//110550085
//Subject:     CO project 2 - ALU
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
module ALU(
    src1_i,
	src2_i,
	ctrl_i,
	result_o,
	zero_o
	);
     
//I/O ports
input signed [32-1:0]  src1_i;
input signed [32-1:0]  src2_i;
input  [4-1:0]   ctrl_i;

output [32-1:0]	 result_o;
output           zero_o;

//Internal signals
reg    [32-1:0]  result_o;
wire             zero_o;

//Parameter
assign zero_o=(result_o==0)?1:0;
//Main function
always@(ctrl_i, src1_i, src2_i)begin
    if(ctrl_i==0)begin
        result_o=src1_i&src2_i;
    end
    else if (ctrl_i==1)begin
        result_o=src1_i|src2_i;
    end
    else if (ctrl_i==2)begin
        result_o=src1_i+src2_i;
    end
    else if (ctrl_i==6)begin
        result_o=src1_i-src2_i;
    end
    else if (ctrl_i==7)begin
        if(src1_i<src2_i)begin
            result_o=1;
        end
        else begin
            result_o=0;
        end
    end
    else begin
        result_o=0;
    end

end

endmodule





                    
                    