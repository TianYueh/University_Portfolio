module	t_Lab2_4_bit_sub();
	wire	[3:0] D1, D2, D3, D4; 
	wire	bout1, bout2, bout3, bout4;
	reg	[3:0] A, B;
	reg     bin;
	
	//instantiate device under test
	Lab2_4_bit_RBS	  	   M1(A, B, bin, D1, bout1);
	Lab2_4_bit_BLS_gatelevel   M2(A, B, bin, D2, bout2);
	Lab2_4_bit_BLS_dataflow    M3(A, B, bin, D3, bout3);
	Lab2_4_bit_BLS_behavioral  M4(A, B, bin, D4, bout4);
	
	//apply inputs one at a time
	initial	begin
		    A=4'b0000; B=4'b1100; bin=1'b1;
		#50 A=4'b0001; B=4'b0010; bin=1'b1;
		#50 A=4'b0011; B=4'b0110; bin=1'b1;
		#50 A=4'b0101; B=4'b1011; bin=1'b0;
		#50 A=4'b0111; B=4'b1010; bin=1'b1;
		#50 A=4'b1000; B=4'b0001; bin=1'b0;
		#50 A=4'b1011; B=4'b0110; bin=1'b0;
		#50 A=4'b1111; B=4'b1111; bin=1'b1;
	end
	initial #450 $finish;

	//dump the result of simulation
	initial begin
		$dumpfile("Lab2_4_bit_sub.vcd");
		$dumpvars;
	end
endmodule