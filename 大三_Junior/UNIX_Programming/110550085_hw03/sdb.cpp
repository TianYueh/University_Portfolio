#include <bits/stdc++.h>
#include <capstone/capstone.h>
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

pid_t pid;
bool isLoaded = false;
string cmd;
vector<string> tokens;
Elf64_Ehdr ehdr; // ELF header
Elf64_Shdr txt_shdr; // .text section header
int file_size = 0;
char* code = NULL;

map <unsigned long long, long> breakpoints;
bool isSiHitBreakpoint = false;
int breakpointnum = 0;
vector<pair<int, unsigned long long>> breakpointscheck;

void load(string execfile) {
    //printf("** try to load program: %s\n", execfile.c_str());
    FILE* fp = fopen(execfile.c_str(), "rb");
    if (fp == NULL) {
		perror("Cannot Open Binary File");
        exit(0);
	}

    isLoaded = true;

    char strtable[1000];
    bzero(strtable, 1000);
    Elf64_Shdr str_shdr, shdr;
    fread(&ehdr, 1, sizeof(ehdr), fp);

    int str_shdr_offset = 0;
    str_shdr_offset = ehdr.e_shstrndx * sizeof(Elf64_Shdr) + ehdr.e_shoff;
    fseek(fp, str_shdr_offset, SEEK_SET);
    fread(&str_shdr, 1, sizeof(Elf64_Shdr), fp);
    fseek(fp, str_shdr.sh_offset, SEEK_SET);
    fread(strtable, str_shdr.sh_size, sizeof(char), fp);
    fseek(fp, ehdr.e_shoff, SEEK_SET);
    for(int i = 0; i < ehdr.e_shnum; i++) {
        fread(&shdr, 1, sizeof(Elf64_Shdr), fp);
        if (strcmp(&strtable[shdr.sh_name], ".text") == 0) {
            txt_shdr = shdr;
            break;
        }
    }
    printf("** program \'%s\' loaded. entry point 0x%lx.\n", execfile.c_str(), ehdr.e_entry);

    pid = fork();
    if(pid < 0) {
        perror("fork");
        exit(0);
    }
    else if(pid == 0) {
        ptrace(PTRACE_TRACEME, 0, NULL, NULL);
        char* args[] = {NULL};
        if(execvp(execfile.c_str(), args) < 0){
            perror("Child execvp error");
            exit(0);
        }
    }
    else {
        int status = 0;
        waitpid(pid, &status, 0);
        ptrace(PTRACE_SETOPTIONS, pid, 0, PTRACE_O_EXITKILL);
    }
}

bool in_text_section(unsigned long addr){
    if(addr >= txt_shdr.sh_addr && (addr < (txt_shdr.sh_addr + txt_shdr.sh_size))){
        return true;
    }
    else{
        return false;
    }
}


void Disassemble(long addr){
    if(!in_text_section(addr)){
        printf("** the address is out of the .text section.\n");
        return;
    }

    long offset = addr - txt_shdr.sh_addr + txt_shdr.sh_offset;
    char* current_code;
    current_code = code + offset;
    csh handle;
    cs_insn *insn;
    size_t count = 0;

    if(cs_open(CS_ARCH_X86, CS_MODE_64, &handle) != CS_ERR_OK){
        printf("ERROR: Failed to initialize engine!\n");
        return;
    }
    count = cs_disasm(handle, (uint8_t*)current_code, (size_t)file_size, addr, 5, &insn);
    if(count <= 0){
        perror("ERROR: Failed to disassemble given code!\n");
        exit(0);
    }

    for(int i = 0 ; i < (int)count; i++){
        //printf("0x%lx: \t%s\t\t%s\n", insn[i].address, insn[i].mnemonic, insn[i].op_str);
        if(in_text_section(insn[i].address)){
            //printf("0x%lx: \t%s\t\t%s\n", insn[i].address, insn[i].mnemonic, insn[i].op_str);
            stringstream ss;
            for(int j = 0; j < insn[i].size; j++){
                ss << setw(2) << right << setfill('0') << hex << (int)insn[i].bytes[j] << " ";
            }
            cout << hex <<right << setw(12) << insn[i].address << ": " << left << setw(32) << ss.str() << left << setw(9) << insn[i].mnemonic << left << setw(9) << insn[i].op_str << endl;
        }
        else{
            printf("** the address is out of the range of the text section.\n");
            break;
        }
    }

    cs_free(insn, count);
    cs_close(&handle);
    return;

}




void breakpoint(unsigned long addr){
    unsigned long new_code;
    new_code = ptrace(PTRACE_PEEKTEXT, pid, addr, NULL);

    if((new_code & 0xff) != 0xcc){
        if(ptrace(PTRACE_POKETEXT, pid, addr, ((new_code & 0xffffffffffffff00) | 0xcc)) != 0){
            perror("PTRACE_POKETEXT");
            exit(0);
        }
        if(breakpoints.find(addr) == breakpoints.end()){
            breakpoints[addr] = new_code;
            breakpointscheck.push_back(make_pair(breakpointnum, addr));
            breakpointnum++;
        }
    }
}

void si(){
    isSiHitBreakpoint = false;
    struct user_regs_struct regs;
    if(ptrace(PTRACE_GETREGS, pid, NULL, &regs) == -1){
        perror("PTRACE_GETREGS");
        exit(0);
    }
    for(auto &bp : breakpoints){
        if(bp.first > regs.rip){
            breakpoint(bp.first);
        }
    }
    if(ptrace(PTRACE_SINGLESTEP, pid, NULL, NULL) == -1){
        perror("PTRACE_SINGLESTEP");
        exit(0);
    }
    int status;
    waitpid(pid, &status, 0);
    if(WIFEXITED(status)){
        cout<<"** the target program terminated.\n";
        exit(0);
    }
    if(WIFSTOPPED(status)){
        if(WSTOPSIG(status) == SIGTRAP){
            if(ptrace(PTRACE_GETREGS, pid, NULL, &regs) == -1){
                perror("PTRACE_GETREGS");
                exit(0);
            }
            for(auto &bp : breakpoints){
                if(bp.first == regs.rip - 1 || bp.first == regs.rip){
                    for(auto &bp : breakpoints){
                        if(bp.first == regs.rip - 1 || bp.first == regs.rip){
                            isSiHitBreakpoint = true;
                            cout<<"** hit a breakpoint at 0x"<<hex<<bp.first<<"."<<endl;
                            if(bp.first == regs.rip - 1){
                                Disassemble(regs.rip - 1);
                            }
                            else{
                                Disassemble(regs.rip);
                            }

                            if(ptrace(PTRACE_POKETEXT, pid, bp.first , bp.second) == -1){
                                perror("PTRACE_POKETEXT");
                                exit(0);
                            }

                            if(bp.first == regs.rip - 1){
                                regs.rip -= 1;
                                if(ptrace(PTRACE_SETREGS, pid, NULL, &regs) == -1){
                                    perror("PTRACE_SETREGS");
                                    exit(0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void CmdContinue(){
    struct user_regs_struct regs;
    if(ptrace(PTRACE_GETREGS, pid, NULL, &regs) == -1){
        perror("PTRACE_GETREGS");
        exit(0);
    }
    //cout<<"current rip: 0x"<<hex<<regs.rip<<endl;
    for(auto &bp : breakpoints){
        if(bp.first > regs.rip){
            breakpoint(bp.first);
            //cout<<"Reset breakpoint at 0x"<<hex<<bp.first<<endl;
        }
    }

    if(ptrace(PTRACE_CONT, pid, NULL, NULL) == -1){
        perror("PTRACE_CONT");
        exit(0);
    }

    int status;
    waitpid(pid, &status, 0);
    if(WIFEXITED(status)){
        cout<<"** the target program terminated.\n";
        exit(0);
    }

    if(WIFSTOPPED(status)){
        if(WSTOPSIG(status) == SIGTRAP){
            if(ptrace(PTRACE_GETREGS, pid, NULL, &regs) == -1){
                perror("PTRACE_GETREGS");
                exit(0);
            }

            //cout << "stopped at: 0x" << hex << regs.rip << endl;
            for(auto &bp : breakpoints){
                if(bp.first == regs.rip - 1 || bp.first == regs.rip){
                    isSiHitBreakpoint = true;
                    cout<<"** hit a breakpoint at 0x"<<hex<<bp.first<<"."<<endl;
                    if(bp.first == regs.rip - 1){
                        Disassemble(regs.rip - 1);
                    }
                    else{
                        Disassemble(regs.rip);
                    }
                    
                    if(ptrace(PTRACE_POKETEXT, pid, bp.first , bp.second) == -1){
                        perror("PTRACE_POKETEXT");
                        exit(0);
                    }
                    
                    if(bp.first == regs.rip - 1){
                        regs.rip -= 1;
                        if(ptrace(PTRACE_SETREGS, pid, NULL, &regs) == -1){
                            perror("PTRACE_SETREGS");
                            exit(0);
                        }
                    }
                    isSiHitBreakpoint = false;
                }
            }
        }
    }
}

void infoReg(){
    struct user_regs_struct regs;
    if(ptrace(PTRACE_GETREGS, pid, NULL, &regs) == -1){
        perror("PTRACE_GETREGS");
        exit(0);
    }
    printf("$rax 0x%016llx\t", regs.rax);
    printf("$rbx 0x%016llx\t", regs.rbx);
    printf("$rcx 0x%016llx\n", regs.rcx);
    printf("$rdx 0x%016llx\t", regs.rdx);
    printf("$rsi 0x%016llx\t", regs.rsi);
    printf("$rdi 0x%016llx\n", regs.rdi);
    printf("$rbp 0x%016llx\t", regs.rbp);
    printf("$rsp 0x%016llx\t", regs.rsp);
    printf("$r8  0x%016llx\n", regs.r8);
    printf("$r9  0x%016llx\t", regs.r9);
    printf("$r10 0x%016llx\t", regs.r10);
    printf("$r11 0x%016llx\n", regs.r11);
    printf("$r12 0x%016llx\t", regs.r12);
    printf("$r13 0x%016llx\t", regs.r13);
    printf("$r14 0x%016llx\n", regs.r14);
    printf("$r15 0x%016llx\t", regs.r15);
    printf("$rip 0x%016llx\t", regs.rip);
    printf("$eflags 0x%016llx\n", regs.eflags);
    return;

}

void infoBreakpoint(){
    cout<<"Num\tAddress\t\n";
    for(long unsigned int i = 0; i < breakpointscheck.size(); i++){
        cout<<breakpointscheck[i].first<<"\t0x"<<hex<<breakpointscheck[i].second<<endl;
    }
    return;
}

void deleteBreakpoint(int num){
    auto it = breakpointscheck.begin();
    long long unsigned int tobedeleted = -1;
    if(breakpointscheck.size() == 0){
        cout<<"** breakpoint "<<num<<" does not exist."<<endl;
        return;
    }
    for(long unsigned int i = 0; i < breakpointscheck.size(); i++){
        if(breakpointscheck[i].first == num){
            tobedeleted = breakpointscheck[i].second;
            break;
        }
        if(i == breakpointscheck.size() - 1){
            cout<<"** breakpoint "<<num<<" does not exist."<<endl;
            return;
        }
    }
    for(long unsigned int i = 0; i < breakpointscheck.size(); i++){
        //cout<<"ICI"<<endl;
        if(breakpointscheck[i].first == num){
            breakpointscheck.erase(it + i);

            if(ptrace(PTRACE_POKETEXT, pid, tobedeleted, breakpoints[tobedeleted]) == -1){
                perror("PTRACE_POKETEXT");
                exit(0);
            }
            cout<<"** delete breakpoint "<<num<<"."<<endl;
            //cout<<tobedeleted<<endl;
            break;
        }
    }
    for(auto &bp : breakpoints){
        //cout<<bp.first<<" "<<bp.second<<endl;
        //cout<<bp.first<<" "<<bp.second<<endl;
        if(bp.first == tobedeleted){
            //cout<<"ICI"<<endl;
            breakpoints.erase(bp.first);
            //cout<<"** delete breakpoint "<<num<<"."<<endl;
            break;
        }
    }

}




void patchMemory(unsigned long addr, unsigned long data, unsigned long len) {
    if (len > 8) {
        printf("** the length is too long.\n");
        return;
    }
    printf("** patch memory at address 0x%lx.\n", addr);

    unsigned long new_code = ptrace(PTRACE_PEEKTEXT, pid, addr, NULL);
    if (len == 1) {
        new_code = (new_code & 0xffffffffffffff00) | (data & 0xff);
    } else if (len == 2) {
        new_code = (new_code & 0xffffffffffff0000) | (data & 0xffff);
    } else if (len == 4) {
        new_code = (new_code & 0xffffffff00000000) | (data & 0xffffffff);
    } else if (len == 8) {
        new_code = (new_code & 0x0000000000000000) | (data & 0xffffffffffffffff);
    }

    //cout<<"new_code: "<<new_code<<endl;
    if (ptrace(PTRACE_POKETEXT, pid, addr, new_code) == -1) {
        perror("PTRACE_POKETEXT");
        exit(0);
    }

    // Reload the instruction at the address after patching
    long offset = addr - txt_shdr.sh_addr + txt_shdr.sh_offset;
    if (offset + len > (long unsigned)file_size) {
        printf("** the patch exceeds the code size.\n");
        return;
    }

    memcpy(code + offset, &new_code, len);

}

bool entering_syscall = true;
//int entering_syscall = 0x01;
void SysCall() {
    struct user_regs_struct regs;
    int status;
    

    while (true) {
        if (ptrace(PTRACE_SYSCALL, pid, 0, 0) == -1) {
            perror("PTRACE_SYSCALL");
            exit(0);
        }
        waitpid(pid, &status, 0);
        //ptrace(PTRACE_SETOPTIONS, pid, 0, PTRACE_O_EXITKILL|PTRACE_O_TRACESYSGOOD);

        if (WIFEXITED(status)) {
            cout << "** the target program terminated.\n";
            return;
        }

        if (WIFSTOPPED(status) && WSTOPSIG(status) == SIGTRAP) {
            if (ptrace(PTRACE_GETREGS, pid, NULL, &regs) == -1) {
                perror("PTRACE_GETREGS");
                exit(0);
            }

            // Check if it's a system call
            if (regs.orig_rax != -1) {
                if (entering_syscall) {
                    // Entering a syscall
                    printf("** enter a syscall(%lld) at 0x%llx.\n", regs.orig_rax, regs.rip - 2);
                } else {
                    // Leaving a syscall
                    printf("** leave a syscall(%lld) = %lld at 0x%llx.\n", regs.orig_rax, regs.rax, regs.rip - 2);
                }

                entering_syscall = !entering_syscall;
                //entering_syscall ^= 0x01;
                Disassemble(regs.rip - 2);
                return; // Return control to the debugger
            } else {
                // Check for breakpoint hit
                for (auto &bp : breakpoints) {
                    if (bp.first == regs.rip - 1 || bp.first == regs.rip) {
                        isSiHitBreakpoint = true;
                        printf("** hit a breakpoint at 0x%llx.\n", bp.first);
                        Disassemble(bp.first);

                        // Restore the original instruction
                        if (ptrace(PTRACE_POKETEXT, pid, bp.first, bp.second) == -1) {
                            perror("PTRACE_POKETEXT");
                            exit(0);
                        }

                        // Adjust the instruction pointer
                        if (regs.rip == bp.first + 1) {
                            regs.rip = bp.first;
                            if (ptrace(PTRACE_SETREGS, pid, NULL, &regs) == -1) {
                                perror("PTRACE_SETREGS");
                                exit(0);
                            }
                        }
                        return; // Return control to the debugger
                    }
                }
            }
        }
    }
}







int main(int argc, char* argv[]) { 
    setvbuf(stdout, NULL, _IONBF, 0);

    string exefile;
    if (argc == 2) {
    	exefile = argv[1];
        load(exefile);
        ifstream fin(exefile.c_str(), ios::in | ios::binary);
        fin.seekg(0, fin.end);
        file_size = fin.tellg();
        fin.seekg(0, fin.beg);
        code = (char*)malloc(sizeof(char) * file_size);
        fin.read(code, file_size);
        fin.close();
        Disassemble(ehdr.e_entry);
	}



    while (1) {
        cout << "(sdb) ";
		string str;
        string part;
        tokens.clear();

		getline(cin, str);
        istringstream iss(str);
		iss >> cmd;
        while (iss >> part) {
            tokens.push_back(part);
        }

        if (isLoaded == false) {
            if (cmd == "load") {
                exefile = tokens[0];
                load(tokens[0]);
                ifstream fin(exefile.c_str(), ios::in | ios::binary);
                fin.seekg(0, fin.end);
                file_size = fin.tellg();
                fin.seekg(0, fin.beg);
                code = (char*)malloc(sizeof(char) * file_size);
                fin.read(code, file_size);
                fin.close();

                //cout << "file size: " << file_size << endl;
                Disassemble(ehdr.e_entry);
            }

            else {
				cout << "** please load a program first." << endl;
                continue;
			}
        }

        /*
        for(long unsigned int i = 0; i < tokens.size(); i++){
            cout<<tokens[i]<<endl;
        }
        */
        if(cmd == "break"){
            breakpoint(stoul(tokens[0], nullptr, 16));
            printf("** set a breakpoint at 0x%lx.\n", stoul(tokens[0], nullptr, 16));
            continue;
        }

        if(cmd == "si"){
            si();
            struct user_regs_struct regs;
            if(ptrace(PTRACE_GETREGS, pid, NULL, &regs) == -1){
                perror("PTRACE_GETREGS");
                exit(0);
            }
            if(!isSiHitBreakpoint){
                Disassemble(regs.rip);
            }

            continue;
        }

        if(cmd == "cont"){
            CmdContinue();
            continue;
        }
    
        if(cmd == "info"){
            if(tokens[0] == "reg"){
                infoReg();
            }
            else if(tokens[0] == "break"){
                infoBreakpoint();
            }
            continue;
        }

        if(cmd == "delete"){
            deleteBreakpoint(stoi(tokens[0]));
            //cout<<"** delete breakpoint "<<stoi(tokens[0])<<"."<<endl;
            continue;
        }

        if(cmd == "patch"){
            patchMemory(stoul(tokens[0], nullptr, 16), stoul(tokens[1], nullptr, 16), stoul(tokens[2], nullptr, 16));
            continue;
        }

        if(cmd == "syscall"){
            SysCall();
            continue;
        }

         

        


        

    
	}



}