module	Lab1_gatelevel_UDP(F, A, B, C, D);
	output	F;
	input	A, B, C, D;
	wire	w1, w2, w3, w4, w5;
	
	not G1(w1, B);
	Lab1_UDP G2(w2, A, w1, C);
	and G3(w3, B, C);
	not G4(w4, D);
	Lab1_UDP G5(w5, w3, w4, A);
	or ANS(F, w2, w5);
endmodule