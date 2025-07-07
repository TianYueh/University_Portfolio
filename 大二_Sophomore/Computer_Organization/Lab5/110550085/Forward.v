`timescale 1ns / 1ps
//110550085

module Forward(
    RegWrite_MEM,
    RegWrite_WB,
    rs_EX,
    rt_EX,
    rd_MEM,
    rd_WB,
    ForwardA,
    ForwardB
    );
    
input RegWrite_MEM;
input RegWrite_WB;
input [5-1:0] rs_EX, rt_EX, rd_MEM, rd_WB;
output [2-1:0] ForwardA;
output [2-1:0] ForwardB;

reg [2-1:0] ForwardA, ForwardB;

always@(*)begin
    //ForwardA
    if((RegWrite_MEM==1)&&(rd_MEM!=0)&&(rs_EX==rd_MEM))begin
        ForwardA=2'b01;
    end
    else if((RegWrite_WB==1)&&(rd_WB!=0)&&(rs_EX==rd_WB))begin
        ForwardA=2'b10;
    end
    else begin
        ForwardA=2'b00;
    end
    
    //ForwardB
    if((RegWrite_MEM==1)&&(rd_MEM!=0)&&(rt_EX==rd_MEM))begin
        ForwardB=2'b01;
    end
    else if((RegWrite_WB==1)&&(rd_WB!=0)&&(rt_EX==rd_WB))begin
        ForwardB=2'b10;
    end
    else begin
        ForwardB=2'b00;
    end


end
    
endmodule