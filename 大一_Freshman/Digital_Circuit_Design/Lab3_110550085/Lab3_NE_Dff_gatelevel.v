module	Lab3_NE_Dff_gatelevel(input D, clock, output Q, Qb);
	wire w1, w2, w3, ul;
	
	Lab3_SR_Latch_gatelevel G1(clock, w3, ul, w1);
	nor	#(10) (w2, clock, w1, w3);
	nor	#(10) (w3, D, w2);
	Lab3_SR_Latch_gatelevel G2(w2, w1, Q, Qb);
	
	
endmodule