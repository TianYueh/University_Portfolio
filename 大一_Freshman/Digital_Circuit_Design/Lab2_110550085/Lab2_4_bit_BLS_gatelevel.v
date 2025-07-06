module	Lab2_4_bit_BLS_gatelevel (input [3:0] A, B, input bin, output [3:0] D, output bout);
	wire w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, p0, p1, p2, p3, g0, g1, g2, g3, b1, b2, b3, not_a0, not_a1, not_a2, not_a3;

	//defining.
	xnor	#(4) (p0, A[0], B[0]);
	xnor	#(4) (p1, A[1], B[1]);
	xnor	#(4) (p2, A[2], B[2]);
	xnor	#(4) (p3, A[3], B[3]);
	not	     (not_a0, A[0]);
	not	     (not_a1, A[1]);
	not	     (not_a2, A[2]);
	not	     (not_a3, A[3]);
	and	#(2) (g0, not_a0, B[0]);
	and	#(2) (g1, not_a1, B[1]);
	and	#(2) (g2, not_a2, B[2]);
	and	#(2) (g3, not_a3, B[3]);

	//let's go.
	and	#(2) (w0, p0, bin);
	or	#(2) (b1, g0, w0);

	and	#(2) (w1, p1, g0);
	and	#(2) (w2, p1, p0, bin);
	or	#(2) (b2, g1, w1, w2);

	and	#(2) (w3, p2, g1);
	and	#(2) (w4, p2, p1, g0);
	and	#(2) (w5, p2, p1, p0, bin);
	or	#(2) (b3, g2, w3, w4, w5);

	and	#(2) (w6, p3, g2);
	and	#(2) (w7, p3, p2, g1);
	and	#(2) (w8, p3, p2, p1, g0);
	and	#(2) (w9, p3, p2, p1, p0, bin);
	or	#(2) (bout, g3, w6, w7, w8, w9);

	xnor	#(4) (D[0], p0, bin);
	xnor	#(4) (D[1], p1, b1);
	xnor	#(4) (D[2], p2, b2);
	xnor	#(4) (D[3], p3, b3);
	//omg.

endmodule