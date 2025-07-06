module	Lab2_converter_4bit_behavioral (input [3:0] E, output reg [3:0] B, output reg v);

	always @(E) begin
		if(E<13&&E>2) v=1;
		else v=0;
		B=E-3;
	end

endmodule