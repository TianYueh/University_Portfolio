module	Lab2_4_bit_BLS_dataflow (input [3:0] A, B, input bin, output [3:0] D, output bout);
	wire w0, w1, w2, w3, p0, p1, p2, p3, g0, g1, g2, g3, b1, b2, b3, not_a0, not_a1, not_a2, not_a3;

	assign	p0 = ((!A[0])&&(!B[0]))||(A[0]&&B[0]),
		g0 = (!A[0])&&B[0],
		w0 = p0&&bin,
		b1 = g0||w0,
		p1 = ((!A[1])&&(!B[1]))||(A[1]&&B[1]),
		g1 = (!A[1])&&B[1],
		w1 = p1&&b1,
		b2 = g1||w1,
		p2 = ((!A[2])&&(!B[2]))||(A[2]&&B[2]),
		g2 = (!A[2])&&B[2],
		w2 = p2&&b2,
		b3 = g2||w2,
		p3 = ((!A[3])&&(!B[3]))||(A[3]&&B[3]),
		g3 = (!A[3])&&B[3],
		w3 = p3&&b3,
		bout = g3||w3,
		D[0] = ((!p0)&&(!bin))||(p0&&bin),
		D[1] = ((!p1)&&(!b1))||(p1&&b1),
		D[2] = ((!p2)&&(!b2))||(p2&&b2),
		D[3] = ((!p3)&&(!b3))||(p3&&b3);
	
endmodule