`timescale 1ns/1ps
// 110550085
module alu_top_overdetect(
    /* input */
    src1,       //1 bit, source 1 (A)
    src2,       //1 bit, source 2 (B)
    less,       //1 bit, less
    A_invert,   //1 bit, A_invert
    B_invert,   //1 bit, B_invert
    cin,        //1 bit, carry in
    operation,  //2 bit, operation
    /* output */
    result,     //1 bit, result
    cout,        //1 bit, carry out
    set   //overflow detection
);

/*==================================================================*/
/*                          input & output                          */
/*==================================================================*/

input src1;
input src2;
input less;
input A_invert;
input B_invert;
input cin;
input [1:0] operation;

output result;
output cout;
output set;

/*==================================================================*/
/*                            reg & wire                            */
/*==================================================================*/

reg result, cout;
wire src11, src21;
wire set;

/*==================================================================*/
/*                              design                              */
/*==================================================================*/


assign src11=A_invert^src1;
assign src21=B_invert^src2;
assign set=src11^src21^cin;

always@(*) begin
    if(operation==2'b00)begin
        result=src11&src21;//and operation
        cout=0;
    end
    else if(operation==2'b01)begin
        result=src11|src21;//or operation
        cout=0;
    end
    else if(operation==2'b10)begin
        //full adder
        result=src11^src21^cin;
        if(src11==1&&src21==1)begin
            cout=1;
        end
        else if(src11==1&&cin==1)begin
            cout=1;
        end
        else if(src21==1&&cin==1)begin
            cout=1;
        end
        else begin
            cout=0;
        end
    end
    else if(operation==2'b11)begin
        //set less than
        result=less;
        if(src11==1&&src21==1)begin
            cout=1;
        end
        else if(src11==1&&cin==1)begin
            cout=1;
        end
        else if(src21==1&&cin==1)begin
            cout=1;
        end
        else begin
            cout=0;
        end
    end
    
end



endmodule