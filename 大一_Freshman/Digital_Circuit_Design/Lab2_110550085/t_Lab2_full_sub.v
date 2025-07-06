module	t_Lab2_full_sub();
	wire	diff, bout;
	reg	a, b, bin;
	
	//instantiate device under test
	Lab2_full_sub		   M1(a, b, bin, diff, bout);
	
	//apply inputs one at a time
	initial	begin
		    a=1'b0; b=1'b0; bin=1'b0;
		#50 a=1'b0; b=1'b0; bin=1'b1;
		#50 a=1'b0; b=1'b1; bin=1'b0;
		#50 a=1'b0; b=1'b1; bin=1'b1;
		#50 a=1'b1; b=1'b0; bin=1'b0;
		#50 a=1'b1; b=1'b0; bin=1'b1;
		#50 a=1'b1; b=1'b1; bin=1'b0;
		#50 a=1'b1; b=1'b1; bin=1'b1;
	end
	initial #500 $finish;

	//dump the result of simulation
	initial begin
		$dumpfile("Lab2_full_sub.vcd");
		$dumpvars;
	end
endmodule