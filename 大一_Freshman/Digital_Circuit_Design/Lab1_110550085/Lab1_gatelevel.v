module	Lab1_gatelevel(F, A, B, C, D);
	output	F;
	input	A, B, C, D;
	wire	w1, w2, w3, w4, w5, w6, w7;
	
	not		G1(w1, B);
	or		G2(w2, A, w1);
	and		G3(w3, w2, C);
	and		G4(w4, B, C);
	not		G5(w5, D);
	or		G6(w6, w4, w5);
	and		G7(w7, A, w6);
	or		G8(F, w3, w7);
endmodule