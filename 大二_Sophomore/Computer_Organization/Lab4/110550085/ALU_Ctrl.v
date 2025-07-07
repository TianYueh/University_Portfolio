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

always@(*) begin
    //R-format
    if(ALUOp_i==3'b010)begin
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
        //or
        else if(funct_i==6'b100101)begin
            ALUCtrl_o = 4'b0001;
        end
        //slt
        else if(funct_i==6'b101010)begin
            ALUCtrl_o = 4'b0111;
        end
        //xor
        else if(funct_i==6'b100110)begin
            ALUCtrl_o = 4'b1100;
        end
        //mult
        else if(funct_i==6'b011000)begin
            ALUCtrl_o = 4'b1000;
        end
        else begin
            ALUCtrl_o = 4'bxxxx;
        end
    end
    //addi
    else if(ALUOp_i==3'b000)begin
        ALUCtrl_o=4'b0010;
    end
    //beq
    else if(ALUOp_i==3'b001)begin
        ALUCtrl_o=4'b0110;
    end
    //slti
    else if(ALUOp_i==3'b011)begin
        ALUCtrl_o=4'b0111;
    end
    else begin
        ALUCtrl_o=4'bxxxx;
    end
end

endmodule     





                    
                    