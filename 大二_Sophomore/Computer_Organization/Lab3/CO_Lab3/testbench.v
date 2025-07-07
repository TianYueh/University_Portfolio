`timescale 1ns / 1ps

`define CYCLE_TIME 5
`define END_COUNT 150

module testbench;
reg Clk, Start;
integer     count;
integer     handle;
integer     end_count;
// reg [31:0] i;

Simple_Single_CPU CPU(Clk, Start);
  
initial
begin
    handle = $fopen("CO_P3_Result.txt");
	Clk = 0;
    Start = 0;
	count = 0;
    #(`CYCLE_TIME) Start = 1;
    #(`CYCLE_TIME*`END_COUNT) begin	
        $fclose(handle); 
        $stop;
    end
end
  
//Print result to "CO_P3_Result.txt"
always@(posedge Clk) begin
    count = count + 1;
    $display("PC = %d", CPU.PC.pc_out_o);
    $display("Count = %d", count);
	if( count == `END_COUNT) begin
        $display(handle, "\nr0=%d\nr1=%d\nr2=%d\nr3=%d\nr4=%d\nr5=%d\nr6=%d\nr7=%d\nr8=%d\nr9=%d\nr10=%d\nr11=%d\nr12=%d\nr13=%d\nr14=%d\nr15=%d\nr16=%d\nr17=%d\nr18=%d\nr19=%d\nr20=%d\nr21=%d\nr22=%d\nr23=%d\nr24=%d\nr25=%d\nr26=%d\nr27=%d\nr28=%d\nr29=%d\nr30=%d\nr31=%d",
	          CPU.Registers.REGISTER_BANK[0], CPU.Registers.REGISTER_BANK[1], CPU.Registers.REGISTER_BANK[2], CPU.Registers.REGISTER_BANK[3], CPU.Registers.REGISTER_BANK[4], 
			  CPU.Registers.REGISTER_BANK[5], CPU.Registers.REGISTER_BANK[6], CPU.Registers.REGISTER_BANK[7], CPU.Registers.REGISTER_BANK[8], CPU.Registers.REGISTER_BANK[9], 
			  CPU.Registers.REGISTER_BANK[10],CPU.Registers.REGISTER_BANK[11], CPU.Registers.REGISTER_BANK[12], CPU.Registers.REGISTER_BANK[13], CPU.Registers.REGISTER_BANK[14],
              CPU.Registers.REGISTER_BANK[15],CPU.Registers.REGISTER_BANK[16], CPU.Registers.REGISTER_BANK[17], CPU.Registers.REGISTER_BANK[18], CPU.Registers.REGISTER_BANK[19],
              CPU.Registers.REGISTER_BANK[20],CPU.Registers.REGISTER_BANK[21], CPU.Registers.REGISTER_BANK[22], CPU.Registers.REGISTER_BANK[23], CPU.Registers.REGISTER_BANK[24],
              CPU.Registers.REGISTER_BANK[25],CPU.Registers.REGISTER_BANK[26], CPU.Registers.REGISTER_BANK[27], CPU.Registers.REGISTER_BANK[28], CPU.Registers.REGISTER_BANK[29],
              CPU.Registers.REGISTER_BANK[30],CPU.Registers.REGISTER_BANK[31]
			  );
        $fdisplay(handle, "2022_CO_Lab3_P3_Result");
        $fdisplay(handle, "r0=%d\nr1=%d\nr2=%d\nr3=%d\nr4=%d\nr5=%d\nr6=%d\nr7=%d\nr8=%d\nr9=%d\nr10=%d\nr11=%d\nr12=%d\nr13=%d\nr14=%d\nr15=%d\nr16=%d\nr17=%d\nr18=%d\nr19=%d\nr20=%d\nr21=%d\nr22=%d\nr23=%d\nr24=%d\nr25=%d\nr26=%d\nr27=%d\nr28=%d\nr29=%d\nr30=%d\nr31=%d",
	          CPU.Registers.REGISTER_BANK[0], CPU.Registers.REGISTER_BANK[1], CPU.Registers.REGISTER_BANK[2], CPU.Registers.REGISTER_BANK[3], CPU.Registers.REGISTER_BANK[4], 
			  CPU.Registers.REGISTER_BANK[5], CPU.Registers.REGISTER_BANK[6], CPU.Registers.REGISTER_BANK[7], CPU.Registers.REGISTER_BANK[8], CPU.Registers.REGISTER_BANK[9], 
			  CPU.Registers.REGISTER_BANK[10],CPU.Registers.REGISTER_BANK[11], CPU.Registers.REGISTER_BANK[12], CPU.Registers.REGISTER_BANK[13], CPU.Registers.REGISTER_BANK[14],
              CPU.Registers.REGISTER_BANK[15],CPU.Registers.REGISTER_BANK[16], CPU.Registers.REGISTER_BANK[17], CPU.Registers.REGISTER_BANK[18], CPU.Registers.REGISTER_BANK[19],
              CPU.Registers.REGISTER_BANK[20],CPU.Registers.REGISTER_BANK[21], CPU.Registers.REGISTER_BANK[22], CPU.Registers.REGISTER_BANK[23], CPU.Registers.REGISTER_BANK[24],
              CPU.Registers.REGISTER_BANK[25],CPU.Registers.REGISTER_BANK[26], CPU.Registers.REGISTER_BANK[27], CPU.Registers.REGISTER_BANK[28], CPU.Registers.REGISTER_BANK[29],
              CPU.Registers.REGISTER_BANK[30],CPU.Registers.REGISTER_BANK[31]
			  );
        $display(handle, "\nm0=%d\nm1=%d\nm2=%d\nm3=%d\nm4=%d\nm5=%d\nm6=%d\nm7=%d\nm8=%d\nm9=%d\nm10=%d\nm11=%d\nm12=%d\nm13=%d\nm14=%d\nm15=%d\nm16=%d\nm17=%d\nm18=%d\nm19=%d\nm20=%d\nm21=%d\nm22=%d\nm23=%d\nm24=%d\nm25=%d\nm26=%d\nm27=%d\nm28=%d\nm29=%d\nm30=%d\nm31=%d",
	          CPU.Data_Memory.memory[0], CPU.Data_Memory.memory[1], CPU.Data_Memory.memory[2], CPU.Data_Memory.memory[3], CPU.Data_Memory.memory[4], 
			  CPU.Data_Memory.memory[5], CPU.Data_Memory.memory[6], CPU.Data_Memory.memory[7], CPU.Data_Memory.memory[8], CPU.Data_Memory.memory[9], 
			  CPU.Data_Memory.memory[10],CPU.Data_Memory.memory[11], CPU.Data_Memory.memory[12], CPU.Data_Memory.memory[13], CPU.Data_Memory.memory[14],
              CPU.Data_Memory.memory[15],CPU.Data_Memory.memory[16], CPU.Data_Memory.memory[17], CPU.Data_Memory.memory[18], CPU.Data_Memory.memory[19],
              CPU.Data_Memory.memory[20],CPU.Data_Memory.memory[21], CPU.Data_Memory.memory[22], CPU.Data_Memory.memory[23], CPU.Data_Memory.memory[24],
              CPU.Data_Memory.memory[25],CPU.Data_Memory.memory[26], CPU.Data_Memory.memory[27], CPU.Data_Memory.memory[28], CPU.Data_Memory.memory[29],
              CPU.Data_Memory.memory[30],CPU.Data_Memory.memory[31]
			  );
        $fdisplay(handle, "m0=%d\nm1=%d\nm2=%d\nm3=%d\nm4=%d\nm5=%d\nm6=%d\nm7=%d\nm8=%d\nm9=%d\nm10=%d\nm11=%d\nm12=%d\nm13=%d\nm14=%d\nm15=%d\nm16=%d\nm17=%d\nm18=%d\nm19=%d\nm20=%d\nm21=%d\nm22=%d\nm23=%d\nm24=%d\nm25=%d\nm26=%d\nm27=%d\nm28=%d\nm29=%d\nm30=%d\nm31=%d",
	          CPU.Data_Memory.memory[0], CPU.Data_Memory.memory[1], CPU.Data_Memory.memory[2], CPU.Data_Memory.memory[3], CPU.Data_Memory.memory[4], 
			  CPU.Data_Memory.memory[5], CPU.Data_Memory.memory[6], CPU.Data_Memory.memory[7], CPU.Data_Memory.memory[8], CPU.Data_Memory.memory[9], 
			  CPU.Data_Memory.memory[10],CPU.Data_Memory.memory[11], CPU.Data_Memory.memory[12], CPU.Data_Memory.memory[13], CPU.Data_Memory.memory[14],
              CPU.Data_Memory.memory[15],CPU.Data_Memory.memory[16], CPU.Data_Memory.memory[17], CPU.Data_Memory.memory[18], CPU.Data_Memory.memory[19],
              CPU.Data_Memory.memory[20],CPU.Data_Memory.memory[21], CPU.Data_Memory.memory[22], CPU.Data_Memory.memory[23], CPU.Data_Memory.memory[24],
              CPU.Data_Memory.memory[25],CPU.Data_Memory.memory[26], CPU.Data_Memory.memory[27], CPU.Data_Memory.memory[28], CPU.Data_Memory.memory[29],
              CPU.Data_Memory.memory[30],CPU.Data_Memory.memory[31]
			  );
    end
    // $display("Data Memory = %d, %d, %d, %d, %d, %d, %d, %d",CPU.Data_Memory.memory[0], CPU.Data_Memory.memory[1], CPU.Data_Memory.memory[2], CPU.Data_Memory.memory[3], CPU.Data_Memory.memory[4], CPU.Data_Memory.memory[5], CPU.Data_Memory.memory[6], CPU.Data_Memory.memory[7]);
    // $display("Data Memory = %d, %d, %d, %d, %d, %d, %d, %d",CPU.Data_Memory.memory[8], CPU.Data_Memory.memory[9], CPU.Data_Memory.memory[10], CPU.Data_Memory.memory[11], CPU.Data_Memory.memory[12], CPU.Data_Memory.memory[13], CPU.Data_Memory.memory[14], CPU.Data_Memory.memory[15]);
    // $display("Data Memory = %d, %d, %d, %d, %d, %d, %d, %d",CPU.Data_Memory.memory[16], CPU.Data_Memory.memory[17], CPU.Data_Memory.memory[18], CPU.Data_Memory.memory[19], CPU.Data_Memory.memory[20], CPU.Data_Memory.memory[21], CPU.Data_Memory.memory[22], CPU.Data_Memory.memory[23]);
    // $display("Data Memory = %d, %d, %d, %d, %d, %d, %d, %d",CPU.Data_Memory.memory[24], CPU.Data_Memory.memory[25], CPU.Data_Memory.memory[26], CPU.Data_Memory.memory[27], CPU.Data_Memory.memory[28], CPU.Data_Memory.memory[29], CPU.Data_Memory.memory[30], CPU.Data_Memory.memory[31]);
    // $display("Registers");
    // $display("R0 = %d, R1 = %d, R2 = %d, R3 = %d, R4 = %d, R5 = %d, R6 = %d, R7 = %d", CPU.Registers.REGISTER_BANK[ 0], CPU.Registers.REGISTER_BANK[ 1], CPU.Registers.REGISTER_BANK[ 2], CPU.Registers.REGISTER_BANK[ 3], CPU.Registers.REGISTER_BANK[ 4], CPU.Registers.REGISTER_BANK[ 5], CPU.Registers.REGISTER_BANK[ 6], CPU.Registers.REGISTER_BANK[ 7]);
    // $display("R8 = %d, R9 = %d, R10 =%d, R11 =%d, R12 =%d, R13 =%d, R14 =%d, R15 =%d", CPU.Registers.REGISTER_BANK[ 8], CPU.Registers.REGISTER_BANK[ 9], CPU.Registers.REGISTER_BANK[10], CPU.Registers.REGISTER_BANK[11], CPU.Registers.REGISTER_BANK[12], CPU.Registers.REGISTER_BANK[13], CPU.Registers.REGISTER_BANK[14], CPU.Registers.REGISTER_BANK[15]);
    // $display("R16 =%d, R17 =%d, R18 =%d, R19 =%d, R20 =%d, R21 =%d, R22 =%d, R23 =%d", CPU.Registers.REGISTER_BANK[16], CPU.Registers.REGISTER_BANK[17], CPU.Registers.REGISTER_BANK[18], CPU.Registers.REGISTER_BANK[19], CPU.Registers.REGISTER_BANK[20], CPU.Registers.REGISTER_BANK[21], CPU.Registers.REGISTER_BANK[22], CPU.Registers.REGISTER_BANK[23]);
    // $display("R24 =%d, R25 =%d, R26 =%d, R27 =%d, R28 =%d, R29 =%d, R30 =%d, R31 =%d", CPU.Registers.REGISTER_BANK[24], CPU.Registers.REGISTER_BANK[25], CPU.Registers.REGISTER_BANK[26], CPU.Registers.REGISTER_BANK[27], CPU.Registers.REGISTER_BANK[28], CPU.Registers.REGISTER_BANK[29], CPU.Registers.REGISTER_BANK[30], CPU.Registers.REGISTER_BANK[31]);
end

always #(`CYCLE_TIME/2) Clk = ~Clk;	
  
endmodule


