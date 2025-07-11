//110550085
//Subject:     CO project 2 - Sign extend
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
module Sign_Extend(
    data_i,
    data_o
    );
               
//I/O ports
input   [16-1:0] data_i;
output  [32-1:0] data_o;

//Internal Signals
reg     [32-1:0] data_o;


//Sign extended
always@(*)begin
    data_o[16-1:0]=data_i[16-1:0];
    if(data_i[16-1]==1)begin
        data_o[32-1:16]=16'b1111111111111111;
    end
    else begin
        data_o[32-1:16]=16'b0000000000000000;
    end
end
          
endmodule      
     