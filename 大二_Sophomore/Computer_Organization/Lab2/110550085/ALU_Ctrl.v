//110550085
//Subject:     CO project 2 - ALU Controller
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
module ALU_Ctrl(
          funct_i,
          ALUOp_i,
          ALUCtrl_o
          );
          
//I/O ports 
input      [6-1:0] funct_i;
input      [3-1:0] ALUOp_i;

output     [4-1:0] ALUCtrl_o;    
     
//Internal Signals
reg        [4-1:0] ALUCtrl_o;

//Parameter
       
//Select exact operation
always@(*)begin
    //addi
    if(ALUOp_i==0)begin
        ALUCtrl_o = 4'b0010;
    end
    //beq
    else if(ALUOp_i==1)begin
        ALUCtrl_o = 4'b0110;
    end
    //R-format
    else if(ALUOp_i==2)begin
        //add
        if(funct_i==6'b100000)begin
            ALUCtrl_o = 4'b0010;
        end
        //sub
        else if(funct_i==6'b100010)begin
            ALUCtrl_o = 4'b0110;
        end
        //and
        else if(funct_i==6'b100100)begin
            ALUCtrl_o = 4'b0000;
        end
        //of
        else if(funct_i==6'b100101)begin
            ALUCtrl_o = 4'b0001;
        end
        //slt
        else if(funct_i==6'b101010)begin
            ALUCtrl_o = 4'b0111;
        end
        else begin
            ALUCtrl_o = 4'bxxxx;
        end
    end
    //slti
    else if(ALUOp_i==3)begin
        ALUCtrl_o = 4'b0111;
    end
    else begin
        ALUCtrl_o = 4'bxxxx;
    end

end

endmodule     





                    
                    