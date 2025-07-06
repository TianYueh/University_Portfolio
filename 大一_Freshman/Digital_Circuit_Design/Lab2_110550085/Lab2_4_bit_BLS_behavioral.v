module	Lab2_4_bit_BLS_behavioral (input [3:0] A, B, input bin, output reg [3:0] D, output reg bout);
	reg w0, w1, w2, w3, p0, p1, p2, p3, g0, g1, g2, g3, b1, b2, b3, not_a0, not_a1, not_a2, not_a3;

	always @(A, B, bin)
		begin
		if(((!A[0])&&(!B[0]))||(A[0]&&B[0])) p0 = 1;
		else p0 = 0;
		if((!A[0])&&B[0]) g0 = 1;
		else g0 = 0;
		if(p0&&bin) w0 = 1;
		else w0 = 0;
		if(g0||w0) b1 = 1;
		else b1 = 0;
		if(((!A[1])&&(!B[1]))||(A[1]&&B[1])) p1 = 1;
		else p1 = 0;
		if((!A[1])&&B[1]) g1 = 1;
		else g1 = 0;
		if(p1&&b1) w1 = 1;
		else w1 = 0;
		if(g1||w1) b2 = 1;
		else b2 = 0;
		if(((!A[2])&&(!B[2]))||(A[2]&&B[2])) p2 = 1;
		else p2 = 0;
		if((!A[2])&&B[2]) g2 = 1;
		else g2 = 0;
		if(p2&&b2) w2 = 1;
		else w2 = 0;
		if(g2||w2) b3 = 1;
		else b3 = 0;
		if(((!A[3])&&(!B[3]))||(A[3]&&B[3])) p3 = 1;
		else p3 = 0;
		if((!A[3])&&B[3]) g3 = 1;
		else g3 = 0;
		if(p3&&b3) w3 = 1;
		else w3 = 0;
		if(g3||w3) bout = 1;
		else bout = 0;
		if(((!p0)&&(!bin))||(p0&&bin)) D[0] = 1;
		else D[0] = 0;
		if(((!p1)&&(!b1))||(p1&&b1)) D[1] = 1;
		else D[1] = 0;
		if(((!p2)&&(!b2))||(p2&&b2)) D[2] = 1;
		else D[2] = 0;
		if(((!p3)&&(!b3))||(p3&&b3)) D[3] = 1;
		else D[3] = 0;
	end
	
endmodule