`timescale 1ns / 1ps
//110550085

module Hazard(
    MemRead,
    instr_i,
    rt_IDEX,
    Branch,
    new,
    flush_IFID,
    flush_IDEX,
    flush_EXMEM
    );
    
input MemRead;
input Branch;
input [16-1:0] instr_i;
input [5-1:0] rt_IDEX;
output new;
output flush_IFID, flush_IDEX, flush_EXMEM;

reg new;
reg flush_IFID, flush_IDEX, flush_EXMEM;

always@(*) begin
    //Branch
    if(Branch)begin
        new=1;
        flush_IFID=1;
        flush_IDEX=1;
        flush_EXMEM=1;
    end
    else begin
        if(MemRead==1)begin
            if((instr_i[9:5]==rt_IDEX)||(instr_i[4:0]==rt_IDEX)&&instr_i[15:10]!=6'b001000)begin
                new=0;
                flush_IFID=0;
                flush_IDEX=1;
                flush_EXMEM=0;
            end
            else begin
                new=1;
                flush_IFID=0;
                flush_IDEX=0;
                flush_EXMEM=0;
            end
        end
        else begin
            new=1;
            flush_IFID=0;
            flush_IDEX=0;
            flush_EXMEM=0;
        end
    
    end

end



endmodule
