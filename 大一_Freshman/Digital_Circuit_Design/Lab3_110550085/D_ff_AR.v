module	D_ff_AR (input D, clock, reset, output reg Q);
	
	always @ (posedge clock, negedge reset)
		if(!reset) Q<=1'b0;
		else Q<=D;
	
	
endmodule