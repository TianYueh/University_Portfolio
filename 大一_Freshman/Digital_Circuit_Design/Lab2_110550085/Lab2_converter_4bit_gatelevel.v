module	Lab2_converter_4bit_gatelevel (input [3:0] E, output [3:0] B, output v);
	wire w0, w1, w2, w3, w4, w5, w6, w7, e0_not, e1_not, e2_not, e3_not;

	not	(B[0], E[0]);
	xor	(B[1], E[1], E[0]);
	not	(e3_not, E[3]);
	not	(e2_not, E[2]);
	not	(e1_not, E[1]);
	not	(e0_not, E[0]);
	and	(w0, e2_not, e0_not);
	and	(w1, E[3], e1_not, E[0]);
	and	(w2, E[2], E[1], E[0]);
	or	(B[2], w0, w1, w2);
	and	(w3, E[3], E[2]);
	and	(w4, E[3], E[1], E[0]);
	or	(B[3], w3, w4);
	xor	(w5, E[3], E[2]);
	and	(w6, E[3], e1_not, e0_not);
	and	(w7, e3_not, E[1], E[0]);
	or	(v, w5, w6, w7);
	
endmodule