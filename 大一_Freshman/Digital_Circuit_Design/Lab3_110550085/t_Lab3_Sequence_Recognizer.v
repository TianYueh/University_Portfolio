module	t_Lab3_Sequence_Recognizer();
	wire	z1, z2;
	reg	x, clock, reset;
	
	//instantiate device under test
	Lab3_Sequence_Recognizer_state_diagram   M1(x, clock, reset, z1);
	Lab3_Sequence_Recognizer_structure	 M2(x, clock, reset, z2);
	
	//apply inputs one at a time

	initial begin
		reset=0;
		clock=0;
		#5 reset=1;
		forever #5 clock=~clock;
	end

	initial	begin
		x=0;
		#15 x=0;
		#10 x=1;
		#10 x=1;
		#10 x=1;
		#10 x=0;
		#10 x=1;
		#10 x=0;
		#10 x=1;
		#10 x=0;
		#10 x=0;
		#10 x=1;
		#10 x=0;
		#10 x=1;
	end

	initial #200 $finish;

	//dump the result of simulation
	initial begin
		$dumpfile("Lab3_Sequence_Recognizer.vcd");
		$dumpvars;
	end
endmodule