module	Lab3_Sequence_Recognizer_structure (input x, clock, reset, output z);
	wire A, B, C, DA, DB, DC;
	
	
	assign DA=(A&(~x))||((~C)&x)||(B&C&(~x));
	assign DB=(B&(~x))||(C&x)||(A&B)||(A&C);
	assign DC=((~A)&(~C)&(~x))||((~B)&(~C)&(~x))||((~A)&(~B)&(~x));
	
	assign z=(~A)&B&(~C)&(~x);

	D_ff_AR M_A(DA, clock, reset, A);
	D_ff_AR M_B(DB, clock, reset, B);
	D_ff_AR M_C(DC, clock, reset, C);
	
	
endmodule