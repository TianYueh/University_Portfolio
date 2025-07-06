module	Lab2_4_bit_RBS(input [3:0] A, B, input bin, output [3:0] D, output bout);
	wire w1, w2, w3;

	Lab2_full_sub		G1(A[0], B[0], bin, D[0], w1);
	Lab2_full_sub		G2(A[1], B[1], w1, D[1], w2);
	Lab2_full_sub		G3(A[2], B[2], w2, D[2], w3);
	Lab2_full_sub		G4(A[3], B[3], w3, D[3], bout);
	
endmodule