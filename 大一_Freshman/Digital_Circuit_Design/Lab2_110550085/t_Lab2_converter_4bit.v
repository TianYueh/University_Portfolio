module	t_Lab2_converter_4bit();
	wire	[3:0] B1, B2, B3, B4; 
	wire	v1, v2, v3, v4;
	reg	[3:0] E;
	
	//instantiate device under test
	Lab2_converter_4bit_gatelevel  M1(E, B1, v1);
	Lab2_converter_4bit_dataflow   M2(E, B2, v2);
	Lab2_converter_4bit_behavioral M3(E, B3, v3);
	
	//apply inputs one at a time
	initial	begin
		    E=4'b0000; 
		#50 E=4'b0001;
		#50 E=4'b0010;
		#50 E=4'b0011;
		#50 E=4'b0100;
		#50 E=4'b0101;
		#50 E=4'b0110;
		#50 E=4'b0111;
		#50 E=4'b1000;
		#50 E=4'b1001;
		#50 E=4'b1010;
		#50 E=4'b1011;
		#50 E=4'b1100;
		#50 E=4'b1101;
		#50 E=4'b1110;
		#50 E=4'b1111;
		
	end
	initial #850 $finish;

	//dump the result of simulation
	initial begin
		$dumpfile("Lab2_converter_4bit.vcd");
		$dumpvars;
	end
endmodule