module	Lab3_SR_Latch_gatelevel (input S, R, output Q, Qb);
	
	nor	#(10) (Q, R, Qb);
	nor	#(10) (Qb, S, Q);
	
	
endmodule