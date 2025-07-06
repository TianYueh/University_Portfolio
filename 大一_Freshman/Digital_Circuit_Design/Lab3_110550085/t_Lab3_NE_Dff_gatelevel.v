module	t_Lab3_NE_Dff_gatelevel();
	wire	Q, Qb;
	reg	D, clock;
	
	//instantiate device under test
	Lab3_NE_Dff_gatelevel	   M1(D, clock, Q, Qb);
	
	//apply inputs one at a time

	initial begin
		clock=0;
		forever #20 clock=~clock;
	end

	initial	fork
		D=0;
		#55 D=1;
		#97 D=0;
		#137 D=1;
		#165 D=0;
		#195 D=1;
		#255 D=0;
	join
	initial #350 $finish;

	//dump the result of simulation
	initial begin
		$dumpfile("Lab3_NE_Dff_gatelevel.vcd");
		$dumpvars;
	end
endmodule