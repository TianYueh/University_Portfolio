#include "codegen/CodeGenerator.hpp"
#include "visitor/AstNodeInclude.hpp"

#include <algorithm>
#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <iostream>


CodeGenerator::CodeGenerator(const std::string &source_file_name,
                             const std::string &save_path,
                             const SymbolManager *const p_symbol_manager)
    : m_symbol_manager_ptr(p_symbol_manager),
      m_source_file_path(source_file_name) {
    // FIXME: assume that the source file is always xxxx.p
    const auto &real_path =
        save_path.empty() ? std::string{"."} : save_path;
    auto slash_pos = source_file_name.rfind("/");
    auto dot_pos = source_file_name.rfind(".");
    assert(dot_pos != std::string::npos && source_file_name[dot_pos+1] == 'p' &&
           "file not recognized");

    if (slash_pos != std::string::npos) {
        ++slash_pos;
    } else {
        slash_pos = 0;
    }
    auto output_file_path{
        real_path + "/" +
        source_file_name.substr(slash_pos, dot_pos - slash_pos) + ".S"};
    m_output_file.reset(fopen(output_file_path.c_str(), "w"));
    assert(m_output_file.get() && "Failed to open output file");

    localAddr = 8;
    p_id = 0;
    label = 1;

    isMain = true;
    isGlobalConst = false;
    isLvalue = false;
    isIf = false;
    isWhile = false;
    isFor = false;
    isBranch = false;
    isFuncinvocation = false;
    isAssigninFor = false;


}

static void dumpInstructions(FILE *p_out_file, const char *format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(p_out_file, format, args);
    va_end(args);
}

void CodeGenerator::stackPush(const std::string p_name) {
    if (addr_et_stack.find(p_name) == addr_et_stack.end()) {
        std::stack<int> temp;
        temp.push(localAddr);
        addr_et_stack[p_name] = temp;
        return;
    }
    addr_et_stack[p_name].push(localAddr);
}

void CodeGenerator::stackPop(const std::string p_name) {
    if(addr_et_stack[p_name].size() == 1){
        addr_et_stack[p_name].pop();
        addr_et_stack.erase(p_name);
        return;
    }
    addr_et_stack[p_name].pop();
}


void CodeGenerator::visit(ProgramNode &p_program) {
    // Generate RISC-V instructions for program header
    // clang-format off
    constexpr const char *const riscv_assembly_file_prologue =
        "    .file \"%s\"\n"
        "    .option nopic\n"
        ".section    .text\n"
        "    .align 2\n";
    // clang-format on
    dumpInstructions(m_output_file.get(), riscv_assembly_file_prologue,
                     m_source_file_path.c_str());

    // Reconstruct the hash table for looking up the symbol entry
    // Hint: Use symbol_manager->lookup(symbol_name) to get the symbol entry.
    m_symbol_manager_ptr->reconstructHashTableFromSymbolTable(
        p_program.getSymbolTable());

    // Visit child nodes
    p_program.visitChildNodes(*this);

    constexpr const char *const riscv_assembly_file_epilogue =
        "# in the function epilogue\n"
        "   lw ra, 124(sp)      # load return address saved in the current stack\n"
        "   lw s0, 120(sp)      # move frame pointer back to the bottom of the last stack\n"
        "   addi sp, sp, 128    # move stack pointer back to the top of the last stack\n"
        "   jr ra               # jump back to the caller function\n"
        "   .size main, .-main\n\n";

    dumpInstructions(m_output_file.get(), riscv_assembly_file_epilogue);

    /*
    auto visit_ast_node = [&](auto &ast_node) { ast_node->accept(*this); };
    for_each(p_program.getDeclNodes().begin(), p_program.getDeclNodes().end(),
             visit_ast_node);
    for_each(p_program.getFuncNodes().begin(), p_program.getFuncNodes().end(),
             visit_ast_node);

    const_cast<CompoundStatementNode &>(p_program.getBody()).accept(*this);
    */

    if(p_program.getSymbolTable() != nullptr){
        const auto &entries = p_program.getSymbolTable()->getEntries();
        for(const auto &entry :entries){
            stackPop(entry->getName());
        }
    }

    // Remove the entries in the hash table
    m_symbol_manager_ptr->removeSymbolsFromHashTable(p_program.getSymbolTable());
}

void CodeGenerator::visit(DeclNode &p_decl) {
    p_decl.visitChildNodes(*this);
}

void CodeGenerator::visit(VariableNode &p_variable) {
    //Get Symbol Entry
    const SymbolEntry* entry = m_symbol_manager_ptr->lookup(p_variable.getName());
    if(entry == nullptr){
        return;
    }
    const char* name = p_variable.getNameCString();

    //Global variable
    if(entry->getLevel() == 0 && entry->getKind() == SymbolEntry::KindEnum::kVariableKind){
        constexpr const char* const riscv_assembly_file_global_var =
            "# global variable declaration: %s\n" 
            ".comm %s, 4, 4\n\n";

        dumpInstructions(m_output_file.get(), riscv_assembly_file_global_var, name, name);
    }
    //Global constant
    else if(entry->getLevel() == 0 && entry->getKind() == SymbolEntry::KindEnum::kConstantKind){
        constexpr const char* const riscv_assembly_file_global_const =
            "# global constant declaration: %s\n"
            ".section    .rodata\n"
            "   .align 2\n"
            "   .globl %s\n"
            "   .type %s, @object\n"
            "%s:\n"
            "    .word ";
        dumpInstructions(m_output_file.get(), riscv_assembly_file_global_const, name, name, name, name);
        isGlobalConst = true;
        p_variable.visitChildNodes(*this);
        dumpInstructions(m_output_file.get(), "\n\n");
    }
    //Local variable
    else if(entry->getKind() == SymbolEntry::KindEnum::kVariableKind){
        stackPush(entry->getName());
        
        if(entry->getTypePtr()->isScalar()){
            if(!entry->getTypePtr()->isString()){
                localAddr += 4;
            }
        }
    }
    //Loop_var
    else if(entry->getKind() == SymbolEntry::KindEnum::kLoopVarKind){
        stackPush(entry->getName());
        if(entry->getTypePtr()->isScalar()){
            localAddr += 4;
        }
        

    }
    //Local constant
    else if(entry->getKind() == SymbolEntry::KindEnum::kConstantKind){
        stackPush(entry->getName());
        constexpr const char* const riscv_assembly_local_const = 
            "# local constant declaration: %s\n"
            "   addi t0, s0, -%d\n"
            "   addi sp, sp, -4\n"
            "   sw t0, 0(sp)        # push the address to the stack\n";
        localAddr += 4;
        dumpInstructions(m_output_file.get(), riscv_assembly_local_const, name, localAddr);
        p_variable.visitChildNodes(*this);
        constexpr const char* const riscv_assembly_local_const_epilogue =
            "   lw t0, 0(sp)        # pop the value from the stack\n"
            "   addi sp, sp, 4\n"
            "   lw t1, 0(sp)        # pop the address from the stack\n"
            "   addi sp, sp, 4\n"
            "   sw t0, 0(t1)\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_local_const_epilogue);
    }
    //Function parameter
    else if(entry->getKind() == SymbolEntry::KindEnum::kParameterKind){
        stackPush(entry->getName());
        constexpr const char* const riscv_assembly_func_param =
            "   sw a%d, -%d(s0)      # save parameter %s in the local stack\n";
        constexpr const char* const riscv_assembly_func_param_but_larger_than_seven = 
            "   sw s%d, -%d(s0)      # save parameter %s in the local stack\n";
    
        if(p_id <= 7){
            if(p_variable.getTypePtr()->isScalar()){
                localAddr += 4;
                dumpInstructions(m_output_file.get(), riscv_assembly_func_param, p_id, localAddr, name);
            }
        }
        else{
            dumpInstructions(m_output_file.get(), riscv_assembly_func_param_but_larger_than_seven, p_id-7, localAddr+4, name);
        }
        p_id++;
    
    }


}

void CodeGenerator::visit(ConstantValueNode &p_constant_value) {

    const char* const_val = p_constant_value.getConstantValueCString();
    if(isGlobalConst){
        isGlobalConst = false;
        dumpInstructions(m_output_file.get(), "%s", const_val);
    }
    else{
        
        if(p_constant_value.getTypePtr()->isBool()){
            if((std::string)const_val == "true"){
                const_val = "1";
            }
            else{
                const_val = "0";
            }
        }
        if(p_constant_value.getTypePtr()->isBool()){
            constexpr const char* const riscv_assembly_const =
            "   li t0, %s            # load value to register 't0'\n"
            "   addi sp, sp, -4\n"
            "   sw t0, 0(sp)        # push the value to the stack\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_const, const_val);
        }
        if(p_constant_value.getTypePtr()->isInteger()){
            constexpr const char* const riscv_assembly_const =
            "   li t0, %s            # load value to register 't0'\n"
            "   addi sp, sp, -4\n"
            "   sw t0, 0(sp)        # push the value to the stack\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_const, const_val);
        }
        else if(p_constant_value.getTypePtr()->isString()){
            constexpr const char* const riscv_assembly_const =
            "   la t0, %s            # load value to register 't0'\n"
            "   addi sp, sp, -4\n"
            "   sw t0, 0(sp)        # push the value to the stack\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_const, const_val);
        }
    }
}

void CodeGenerator::visit(FunctionNode &p_function) {
    // Reconstruct the hash table for looking up the symbol entry
    m_symbol_manager_ptr->reconstructHashTableFromSymbolTable(
        p_function.getSymbolTable());

    localAddr = 8;
    isMain = false;
    const char* funcName = p_function.getNameCString();
    constexpr const char* const riscv_assembly_func = 
        ".section    .text\n"
        "   .align 2\n"
        "   .globl %s\n"
        "   .type %s, @function\n\n"
        "%s:\n"
        "# in the function prologue\n"
        "   addi sp, sp, -128   # move stack pointer to lower address to allocate a new stack\n"
        "   sw ra, 124(sp)      # save return address of the caller function in the current stack\n"
        "   sw s0, 120(sp)      # save frame pointer of the last stack in the current stack\n"
        "   addi s0, sp, 128    # move frame pointer to the bottom of the current stack\n\n";
    dumpInstructions(m_output_file.get(), riscv_assembly_func, funcName, funcName, funcName);

    p_function.visitChildNodes(*this);

    p_id = 0;
    isMain = true;
    localAddr = 8;
    constexpr const char* const riscv_assembly_func_epilogue =
        "# in the function epilogue\n"
        "   lw ra, 124(sp)      # load return address saved in the current stack\n"
        "   lw s0, 120(sp)      # move frame pointer back to the bottom of the last stack\n"
        "   addi sp, sp, 128    # move stack pointer back to the top of the last stack\n"
        "   jr ra               # jump back to the caller function\n"
        "   .size %s, .-%s\n\n";

    dumpInstructions(m_output_file.get(), riscv_assembly_func_epilogue, funcName, funcName);
    if(p_function.getSymbolTable() != nullptr){
        const auto &entries = p_function.getSymbolTable()->getEntries();
        for(const auto &entry :entries){
            stackPop(entry->getName());
        }
    }
    
    // Remove the entries in the hash table
    m_symbol_manager_ptr->removeSymbolsFromHashTable(p_function.getSymbolTable());
}

void CodeGenerator::visit(CompoundStatementNode &p_compound_statement) {
    // Reconstruct the hash table for looking up the symbol entry
    m_symbol_manager_ptr->reconstructHashTableFromSymbolTable(
        p_compound_statement.getSymbolTable());

    if(isMain){
        isMain = false;
        constexpr const char* const riscv_assembly_main = 
            "   .globl main         # emit symbol 'main' to the global symbol table\n"
            "   .type main, @function\n\n"
            "main:\n"
            "# in the function prologue\n"
            "   addi sp, sp, -128   # move stack pointer to lower address to allocate a new stack\n"
            "   sw ra, 124(sp)      # save return address of the caller function in the current stack\n"
            "   sw s0, 120(sp)      # save frame pointer of the last stack in the current stack\n"
            "   addi s0, sp, 128    # move frame pointer to the bottom of the current stack\n\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_main);
        
    }

    if(isIf){
        dumpInstructions(m_output_file.get(), "L%d:\n", label_id);
    }
    if(isWhile){
        dumpInstructions(m_output_file.get(), "L%d:\n", label_id+1);
    }
    if(isFor){
        constexpr const char* const riscv_assembly_for_epilogue =
            "   lw t0, 0(sp)        # pop the value from the stack\n"
            "   addi sp, sp, 4\n"
            "   lw t1, 0(sp)        # pop the value from the stack\n"
            "   addi sp, sp, 4\n"
            "   bge t1, t0, L%d        # if i >= condition value, exit the loop\n"
            "L%d:\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_for_epilogue, label_id+2, label_id+1);
    }


    p_compound_statement.visitChildNodes(*this);

    if(isIf){
        isIf = false;
        dumpInstructions(m_output_file.get(), "   j L%d                # jump to L%d\nL%d:\n",label_id+2 ,label_id+2 ,label_id+1);
    }
    if(isWhile){
        isWhile = false;
        dumpInstructions(m_output_file.get(), "   j L%d                # jump to L%d\n",label_id ,label_id);
    }

    if(p_compound_statement.getSymbolTable() != nullptr){
        const auto &entries = p_compound_statement.getSymbolTable()->getEntries();
        for(const auto &entry :entries){
            stackPop(entry->getName());
        }
    }

    // Remove the entries in the hash table
    m_symbol_manager_ptr->removeSymbolsFromHashTable(
        p_compound_statement.getSymbolTable());
}

void CodeGenerator::visit(PrintNode &p_print) {
    dumpInstructions(m_output_file.get(), "# print\n");

    p_print.visitChildNodes(*this);

    if(p_print.getTarget().getInferredType()->isInteger()){
        constexpr const char* const riscv_assembly_print = 
            "   lw a0, 0(sp)        # pop the value from the stack to the first argument register 'a0'\n"
            "   addi sp, sp, 4\n"
            "   jal ra, printInt    # call function 'printInt'\n\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_print);
    }
    else if(p_print.getTarget().getInferredType()->isString()){
        constexpr const char* const riscv_assembly_print_string = 
            "   lw a0, 0(sp)        # pop the value from the stack to the first argument register 'a0'\n"
            "   addi sp, sp, 4\n"
            "   jal ra, printString    # call function 'printString'\n\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_print_string);
    }
    


}

void CodeGenerator::visit(BinaryOperatorNode &p_bin_op) {
    dumpInstructions(m_output_file.get(), "\n# binary operator: %s\n", p_bin_op.getOpCString());
    bool fBranch = isBranch;
    isBranch = false;
    p_bin_op.visitChildNodes(*this);

    constexpr const char* const riscv_assembly_push =
        "   addi sp, sp, -4\n"
        "   sw t0, 0(sp)        # push the value to the stack\n";

    constexpr const char* const riscv_assembly_pop_t0_t1 = 
        "   lw t0, 0(sp)        # pop the value from the stack\n"
        "   addi sp, sp, 4\n"
        "   lw t1, 0(sp)        # pop the value from the stack\n"
        "   addi sp, sp, 4\n";

    constexpr const char* const riscv_assembly_branch =
        "   %s t1, t0, L%d      # if t1 %s t0, jump to L%d\n";
    dumpInstructions(m_output_file.get(), riscv_assembly_pop_t0_t1);

    const char* op = p_bin_op.getOpCString();

    if((std::string)op == "+"){
        dumpInstructions(m_output_file.get(), "   add t0, t1, t0      # always save the value in a certain register you choose\n");
        dumpInstructions(m_output_file.get(), riscv_assembly_push);
    }
    else if((std::string)op == "-"){
        dumpInstructions(m_output_file.get(), "   sub t0, t1, t0      # always save the value in a certain register you choose\n");
        dumpInstructions(m_output_file.get(), riscv_assembly_push);
    }
    else if((std::string)op == "*"){
        dumpInstructions(m_output_file.get(), "   mul t0, t1, t0      # always save the value in a certain register you choose\n");
        dumpInstructions(m_output_file.get(), riscv_assembly_push);
    }
    else if((std::string)op == "/"){
        dumpInstructions(m_output_file.get(), "   div t0, t1, t0      # always save the value in a certain register you choose\n");
        dumpInstructions(m_output_file.get(), riscv_assembly_push);
    }
    else if((std::string)op == "mod"){
        dumpInstructions(m_output_file.get(), "   rem t0, t1, t0      # always save the value in a certain register you choose\n");
        dumpInstructions(m_output_file.get(), riscv_assembly_push);
    }
    else if((std::string)op == "or"){
        dumpInstructions(m_output_file.get(), "   or t0, t1, t0      # always save the value in a certain register you choose\n");
        dumpInstructions(m_output_file.get(), riscv_assembly_push);
    }
    else if((std::string)op == "and"){
        dumpInstructions(m_output_file.get(), "   and t0, t1, t0      # always save the value in a certain register you choose\n");
        dumpInstructions(m_output_file.get(), riscv_assembly_push);
    }
    else if((std::string)op == "<="){
        if(isIf && fBranch){
            dumpInstructions(m_output_file.get(), riscv_assembly_branch, "bgt", label_id+1, ">", label_id+1);
        }
        else if(isWhile && fBranch){
            dumpInstructions(m_output_file.get(), riscv_assembly_branch, "bgt", label_id+2, ">", label_id+2);
        }
        else{
            constexpr const char* const riscv_assembly_bool =
                "   sub t0, t1, t0      # always save the value in a certain register you choose\n"
                "   slti t0, t0, 1\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_bool);
            dumpInstructions(m_output_file.get(), riscv_assembly_push);
        }
    }
    else if((std::string)op == ">="){
        if(isIf && fBranch){
            dumpInstructions(m_output_file.get(), riscv_assembly_branch, "blt", label_id+1, "<", label_id+1);
        }
        else if(isWhile && fBranch){
            dumpInstructions(m_output_file.get(), riscv_assembly_branch, "blt", label_id+2, "<", label_id+2);
        }
        else{
            constexpr const char* const riscv_assembly_bool =
                "   sub t0, t1, t0      # always save the value in a certain register you choose\n"
                "   slti t0, t0, 0\n"
                "   slti t0, t0, 1\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_bool);
            dumpInstructions(m_output_file.get(), riscv_assembly_push);
        }
    }
    else if((std::string)op == "<"){
        if(isIf && fBranch){
            dumpInstructions(m_output_file.get(), riscv_assembly_branch, "bge", label_id+1, ">=", label_id+1);
        }
        else if(isWhile && fBranch){
            dumpInstructions(m_output_file.get(), riscv_assembly_branch, "bge", label_id+2, ">=", label_id+2);
        }
        else{
            constexpr const char* const riscv_assembly_bool =
                "   sub t0, t1, t0      # always save the value in a certain register you choose\n"
                "   slti t0, t0, 0\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_bool);
            dumpInstructions(m_output_file.get(), riscv_assembly_push);
        }
    }
    else if((std::string)op == ">"){
        if(isIf && fBranch){
            dumpInstructions(m_output_file.get(), riscv_assembly_branch, "ble", label_id+1, "<=", label_id+1);
        }
        else if(isWhile && fBranch){
            dumpInstructions(m_output_file.get(), riscv_assembly_branch, "ble", label_id+2, "<=", label_id+2);
        }
        else{
            constexpr const char* const riscv_assembly_bool =
                "   sub t0, t1, t0      # always save the value in a certain register you choose\n"
                "   slti t0, t0, 1\n"
                "   slti t0, t0, 1\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_bool);
            dumpInstructions(m_output_file.get(), riscv_assembly_push);
        }
    }
    else if((std::string)op == "="){
        if(isIf && fBranch){
            dumpInstructions(m_output_file.get(), riscv_assembly_branch, "bne", label_id+1, "!=", label_id+1);
        }
        else if(isWhile && fBranch){
            dumpInstructions(m_output_file.get(), riscv_assembly_branch, "bne", label_id+2, "!=", label_id+2);
        }
        else{
            constexpr const char* const riscv_assembly_bool =
                "   sub t0, t1, t0      # always save the value in a certain register you choose\n"
                "   seqz t0, t0\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_bool);
            dumpInstructions(m_output_file.get(), riscv_assembly_push);
        }
    }
    else if((std::string)op == "<>"){
        if(isIf && fBranch){
            dumpInstructions(m_output_file.get(), riscv_assembly_branch, "beq", label_id+1, "=", label_id+1);
        }
        else if(isWhile && fBranch){
            dumpInstructions(m_output_file.get(), riscv_assembly_branch, "beq", label_id+2, "=", label_id+2);
        }
        else{
            constexpr const char* const riscv_assembly_bool =
                "   sub t0, t1, t0      # always save the value in a certain register you choose\n"
                "   snez t0, t0, 1\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_bool);
            dumpInstructions(m_output_file.get(), riscv_assembly_push);
        }
    }
    dumpInstructions(m_output_file.get(), "\n");
    
    
    


}

void CodeGenerator::visit(UnaryOperatorNode &p_un_op) {
    const char* op = p_un_op.getOpCString();
    dumpInstructions(m_output_file.get(), "\n# unary operator: %s\n", op);
    if((std::string)op == "neg"){
        p_un_op.visitChildNodes(*this);
        constexpr const char* const riscv_assembly_unary = 
            "   lw t0, 0(sp)        # pop the value from the stack\n"
            "   addi sp, sp, 4\n"
            "   neg t0, t0\n        # always save the value in a certain register you choose\n"
            "   addi sp, sp, -4\n"
            "   sw t0, 0(sp)        # push the value to the stack\n\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_unary);
    }
    else if((std::string)op == "not"){
        bool fBranch = isBranch;
        isBranch = false;
        p_un_op.visitChildNodes(*this);

        constexpr const char* const riscv_assembly_not =
            "   lw t0, 0(sp)        # pop the value from the stack\n"
            "   addi sp, sp, 4\n"
            "   slti t0, t0, 1\n"
            "   addi sp, sp, -4\n"
            "   sw t0, 0(sp)        # push the value to the stack\n\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_not);

        if(fBranch){
            constexpr const char* const riscv_assembly_not_branch =
                "   lw t1, 0(sp)         # pop the value from the stack\n"
                "   addi sp, sp, 4\n"
                "   li t0, 0\n"
                "   beq t1, t0, L%d      # if t1 == 0, jump to L%d\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_not_branch, label_id+1, label_id+1);
        }
    
    
    
    }
}

void CodeGenerator::visit(FunctionInvocationNode &p_func_invocation) {
    dumpInstructions(m_output_file.get(), "\n# function invocation: %s\n", p_func_invocation.getNameCString());
    isFuncinvocation = true;

    p_func_invocation.visitChildNodes(*this);

    isFuncinvocation = false;
    constexpr const char* const riscv_assembly_pop_a =
        "   lw a%d, 0(sp)        # pop the value from the stack to the argument register a%d\n"
        "   addi sp, sp, 4\n";
    constexpr const char* const riscv_assembly_pop_s =
        "   lw s%d, 0(sp)        # pop the value from the stack to the argument register s%d\n"
        "   addi sp, sp, 4\n";
    
    int s = p_func_invocation.getArguments().size()-1;
    for(int i=s;i>=0;i--){

        if(i<=7){
            if(p_func_invocation.getArguments()[i]->getInferredType()->isScalar()){
                dumpInstructions(m_output_file.get(), riscv_assembly_pop_a, i, i);
            }
        }
        else{
            dumpInstructions(m_output_file.get(), riscv_assembly_pop_s, i-7, i-7);
        }


    }
    constexpr const char*const riscv_assembly_funcinvocation=
    "   jal ra, %s         # call function %s\n"
    "   mv t0, a0          # always move the return value to a certain register you choose\n"
    "   addi sp, sp, -4\n"
    "   sw t0, 0(sp)       # push the value to the stack\n\n\n";

    dumpInstructions(m_output_file.get(), riscv_assembly_funcinvocation, p_func_invocation.getNameCString(), p_func_invocation.getNameCString());
}

void CodeGenerator::visit(VariableReferenceNode &p_variable_ref) {
    const SymbolEntry* entry = m_symbol_manager_ptr->lookup(p_variable_ref.getName());
    if(entry == nullptr){
        return;
    }

    const char* name = p_variable_ref.getNameCString();
    const char* entry_name = entry->getNameCString();
    //Global variable
    if(entry->getLevel() == 0 && entry->getKind() == SymbolEntry::KindEnum::kVariableKind){
        if(isLvalue){
            isLvalue = false;
            constexpr const char* const riscv_assembly_global_var_lvalue_ref =
                "   la t0, %s           # load the address of variable %s\n"
                "   addi sp, sp, -4\n"
                "   sw t0, 0(sp)     # push the address to the stack\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_global_var_lvalue_ref, name, name);   
        }
        else{
            constexpr const char* const riscv_assembly_global_var_rvalue_ref =
                "   la t0, %s\n"
                "   lw t1, 0(t0)        # load the value of %s\n"
                "   mv t0, t1\n"
                "   addi sp, sp, -4\n"
                "   sw t0, 0(sp)     # push the address to the stack\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_global_var_rvalue_ref, name, name);
        }
    }
    //OK
    //Global constant
    else if(entry->getLevel() == 0 && entry->getKind() == SymbolEntry::KindEnum::kConstantKind){
        constexpr const char* const riscv_assembly_global_const_ref =
            "   la t0, %s\n"
            "   lw t1, 0(t0)        # load the value of %s\n"
            "   mv t0, t1\n"
            "   addi sp, sp, -4\n"
            "   sw t0, 0(sp)     # push the value to the stack\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_global_const_ref, name, name);
    }
    //Local variable/loop_var/parameter
    else if(entry->getKind() == SymbolEntry::KindEnum::kVariableKind || entry->getKind() == SymbolEntry::KindEnum::kLoopVarKind || entry->getKind() == SymbolEntry::KindEnum::kParameterKind){
        int addr = addr_et_stack[entry->getName()].top();
        if(isLvalue){
            isLvalue = false;
            if(entry->getTypePtr()->isScalar()){
                if(!entry->getTypePtr()->isString()){
                    constexpr const char* const riscv_assembly_local_var_lvalue_ref =
                        "   addi t0, s0, -%d\n"
                        "   addi sp, sp, -4\n"
                        "   sw t0, 0(sp)     # push the address to the stack\n";
                    dumpInstructions(m_output_file.get(), riscv_assembly_local_var_lvalue_ref, addr+4);
                }
                else{
                    constexpr const char* const riscv_assembly_const_str =
                        "    .section    .rodata\n"
                        "    .align 2\n"
                        "%s:\n"
                        "    .string ";
                    dumpInstructions(m_output_file.get(), riscv_assembly_const_str, entry_name);
                }
            }
        }
        //Rvalue
        else{
            if(entry->getTypePtr()->isScalar()){
                if(entry->getTypePtr()->isString()){
                    constexpr const char* const riscv_assembly_str_rvalue_ref =
                        "   lui t0, %%hi(%s)\n"
                        "   addi t0, t0, %%lo(%s)\n"
                        "   addi sp, sp, -4\n"
                        "   sw t0, 0(sp)        # push the value to the stack\n";
                    dumpInstructions(m_output_file.get(), riscv_assembly_str_rvalue_ref, entry_name, entry_name);
                }
                else{
                    constexpr const char* const riscv_assembly_local_var_rvalue_ref =
                        "   lw t0, -%d(s0)      # load the value of %s\n"
                        "   addi sp, sp, -4\n"
                        "   sw t0, 0(sp)        # push the value to the stack\n";
                    dumpInstructions(m_output_file.get(), riscv_assembly_local_var_rvalue_ref, addr+4, name);
                }
            }


        }
        
        
    }
    //Local constant
    else if(entry->getKind() == SymbolEntry::KindEnum::kConstantKind){
        int addr = addr_et_stack[entry->getName()].top();
        constexpr const char* const riscv_assembly_local_rvalue_const =
            "   lw t0, -%d(s0)      # load the value of %s\n"
            "   addi sp, sp, -4\n"
            "   sw t0, 0(sp)        # push the value to the stack\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_local_rvalue_const, addr+4, name);
    
    }
    //Branch
    if(isBranch){
        isBranch = false;
        constexpr const char* const riscv_assembly_branch =
            "   lw t1, 0(sp)         # pop the value from the stack\n"
            "   addi sp, sp, 4\n"
            "   li t0, 0\n"
            "   beq t1, t0, L%d      # if t1 == 0, jump to L%d\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_branch, label_id+1, label_id+1);
    }

}

void CodeGenerator::visit(AssignmentNode &p_assignment) {
    isLvalue = true;
    dumpInstructions(m_output_file.get(), "\n# variable assignment: %s\n", p_assignment.getLvalue().getNameCString());

    p_assignment.visitChildNodes(*this);

    if(!p_assignment.getLvalue().getInferredType()->isString()){
        constexpr const char* const riscv_assembly_assignment = 
            "   lw t0, 0(sp)        # pop the value from the stack\n"
            "   addi sp, sp, 4\n"
            "   lw t1, 0(sp)        # pop the address from the stack\n"
            "   addi sp, sp, 4\n"
            "   sw t0, 0(t1)\n";

            dumpInstructions(m_output_file.get(), riscv_assembly_assignment);
    }
    if(isAssigninFor){
        isAssigninFor = false;
        int addr = addr_et_stack[p_assignment.getLvalue().getName()].top();
        constexpr const char* const riscv_assembly_assigninfor =
            "L%d:\n"
            "   lw t0, -%d(s0)      # load the value of %s\n"
            "   addi sp, sp, -4\n"
            "   sw t0, 0(sp)        # push the value to the stack\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_assigninfor, label_id, addr+4, p_assignment.getLvalue().getNameCString());
    }


}

void CodeGenerator::visit(ReadNode &p_read) {


    dumpInstructions(m_output_file.get(), "# read\n");
    isLvalue = true;
    p_read.visitChildNodes(*this);
    constexpr const char* const riscv_assembly_read =
        "   jal ra, readInt     # call function 'readInt'\n"
        "   lw t0, 0(sp)        # pop the address from the stack\n"
        "   addi sp, sp, 4\n"
        "   sw a0, 0(t0)        # store the value to the address\n";
    dumpInstructions(m_output_file.get(), riscv_assembly_read);
}

void CodeGenerator::visit(IfNode &p_if) {
    isIf = true;
    isBranch = true;
    baseOfLabel.push(label);
    label+=3;
    label_id = baseOfLabel.top();

    p_if.visitChildNodes(*this);

    isIf = false;
    dumpInstructions(m_output_file.get(), "L%d:\n", label_id+2);
    baseOfLabel.pop();
    if(baseOfLabel.empty()){
        label_id = label;
    }
    else{
        label_id = baseOfLabel.top();
        
    }

}

void CodeGenerator::visit(WhileNode &p_while) {
    baseOfLabel.push(label);
    label+=3;
    label_id = baseOfLabel.top();
    isWhile = true;
    isBranch = true;
    dumpInstructions(m_output_file.get(), "L%d:\n", label_id);

    p_while.visitChildNodes(*this);

    isWhile = false;
    dumpInstructions(m_output_file.get(), "L%d:\n", label_id + 2);
    baseOfLabel.pop();

    if(!baseOfLabel.empty()){
        label_id = baseOfLabel.top();
    }
    else{
        label_id = label;
    }
}

void CodeGenerator::visit(ForNode &p_for) {
    // Reconstruct the hash table for looking up the symbol entry
    m_symbol_manager_ptr->reconstructHashTableFromSymbolTable(
        p_for.getSymbolTable());

    
    isAssigninFor = true;

    baseOfLabel.push(label);
    label+=3;
    label_id = baseOfLabel.top();
    isFor = true;

    p_for.visitChildNodes(*this);

    isFor = false;
    int addr = addr_et_stack[p_for.getInitStmt()->getLvalue().getName()].top();

    constexpr const char* const riscv_assembly_for = 
    "   addi t0, s0, -%d      # load the address of loop variable\n"
    "   addi sp, sp, -4\n"
    "   sw t0, 0(sp)        # push the address to the stack\n"
    "   lw t0, -%d(s0)      # load the value of loop variable\n"
    "   addi sp, sp, -4\n"
    "   sw t0, 0(sp)        # push the value to the stack\n"
    "   li t0, 1\n"
    "   addi sp, sp, -4\n"
    "   sw t0, 0(sp)        # push the value to the stack\n"
    "   lw t0, 0(sp)        # pop the value from the stack\n"
    "   addi sp, sp, 4\n"
    "   lw t1, 0(sp)        # pop the value from the stack\n"
    "   addi sp, sp, 4\n"
    "   add t0, t1, t0      # always save the value in a certain register you choose\n"
    "   addi sp, sp, -4\n"
    "   sw t0, 0(sp)        # push the value to the stack\n"
    "   lw t0, 0(sp)        # pop the value from the stack\n"
    "   addi sp, sp, 4\n"
    "   lw t1, 0(sp)        # pop the address from the stack\n"
    "   addi sp, sp, 4\n"
    "   sw t0, 0(t1)        # save the value to loop variable\n"
    "   j L%d                # jump back to loop condition\n"
    "L%d:\n";

    dumpInstructions(m_output_file.get(), riscv_assembly_for, addr+4, addr+4, label_id, label_id+2);

    baseOfLabel.pop();
    if(!baseOfLabel.empty()){
        label_id = baseOfLabel.top();
    }
    else{
        label_id = label;
    }

    if(p_for.getSymbolTable() != nullptr){
        const auto &entries = p_for.getSymbolTable()->getEntries();
        for(const auto &entry :entries){
            stackPop(entry->getName());
        }
    }

    // Remove the entries in the hash table
    m_symbol_manager_ptr->removeSymbolsFromHashTable(p_for.getSymbolTable());
}

void CodeGenerator::visit(ReturnNode &p_return) {
    p_return.visitChildNodes(*this);
    constexpr const char* const riscv_assembly_return =
        "   lw t0, 0(sp)        # pop the value from the stack\n"
        "   addi sp, sp, 4\n"
        "   mv a0, t0    # load the value to the return value register 'a0'\n\n";
    dumpInstructions(m_output_file.get(), riscv_assembly_return);

}