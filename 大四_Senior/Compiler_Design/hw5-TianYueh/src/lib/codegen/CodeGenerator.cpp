#include "AST/CompoundStatement.hpp"
#include "AST/for.hpp"
#include "AST/function.hpp"
#include "AST/program.hpp"
#include "codegen/CodeGenerator.hpp"
#include "sema/SemanticAnalyzer.hpp"
#include "sema/SymbolTable.hpp"
#include "visitor/AstNodeInclude.hpp"

#include <algorithm>
#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <iostream>
#include <cstring>

CodeGenerator::CodeGenerator(const std::string &source_file_name,
                             const std::string &save_path,
                             std::unordered_map<SemanticAnalyzer::AstNodeAddr,
                                                      SymbolManager::Table>
                                 &&p_symbol_table_of_scoping_nodes)
    : m_symbol_manager(false /* no dump */),
      m_source_file_path(source_file_name),
      m_symbol_table_of_scoping_nodes(std::move(p_symbol_table_of_scoping_nodes)) {
    // FIXME: assume that the source file is always xxxx.p
    const auto &real_path =
        save_path.empty() ? std::string{"."} : save_path;
    auto slash_pos = source_file_name.rfind('/');
    auto dot_pos = source_file_name.rfind('.');

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
}

static void dumpInstructions(FILE *p_out_file, const char *format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(p_out_file, format, args);
    va_end(args);
}

void CodeGenerator::visit(ProgramNode &p_program) {
    // Generate RISC-V instructions for program header
    // clang-format off
    constexpr const char *const riscv_assembly_file_prologue =
        "    .file \"%s\"\n"
        "    .option nopic\n";
    // clang-format on
    dumpInstructions(m_output_file.get(), riscv_assembly_file_prologue,
                     m_source_file_path.c_str());

    // Reconstruct the scope for looking up the symbol entry.
    // Hint: Use m_symbol_manager->lookup(symbol_name) to get the symbol entry.
    m_symbol_manager.pushScope(
        std::move(m_symbol_table_of_scoping_nodes.at(&p_program)));

    auto visit_ast_node = [&](auto &ast_node) { ast_node->accept(*this); };
    for_each(p_program.getDeclNodes().begin(), p_program.getDeclNodes().end(),
             visit_ast_node);
    for_each(p_program.getFuncNodes().begin(), p_program.getFuncNodes().end(),
             visit_ast_node);
    
    constexpr const char *const riscv_assembly_main = 
        ".section    .text\n"
        "    .align 2\n"
        "    .globl main\n"
        "    .type main, @function\n"
        "main:\n";

    dumpInstructions(m_output_file.get(), riscv_assembly_main);
    
    const_cast<CompoundStatementNode &>(p_program.getBody()).accept(*this);
    m_symbol_manager.popScope();

    constexpr const char *const riscv_assembly_file_epilogue =
        "    .size main, .-main\n";

    dumpInstructions(m_output_file.get(), riscv_assembly_file_epilogue);
}

void CodeGenerator::visit(DeclNode &p_decl) {
    p_decl.visitChildNodes(*this);
}

void CodeGenerator::visit(VariableNode &p_variable) {
    SymbolEntry *entry = const_cast<SymbolEntry *>(m_symbol_manager.lookup(p_variable.getName()));
    bool isGlobal = (entry->getLevel() == 0);
    SymbolEntry::KindEnum kind = entry->getKind();
    if (isGlobal){
        if(kind == SymbolEntry::KindEnum::kConstantKind) {
            // Generate RISC-V instructions for global constant variable
            constexpr const char *const riscv_assembly_global_const = 
                ".section    .rodata\n"
                "    .align 2\n"
                "    .globl %s\n"
                "    .type %s, @object\n%s:\n"
                "    .word %s\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_global_const, p_variable.getNameCString(),p_variable.getNameCString(), p_variable.getNameCString(), entry->getAttribute().constant()->getConstantValueCString());
        }
        else if (kind == SymbolEntry::KindEnum::kVariableKind) {
            // Generate RISC-V instructions for global variable
            constexpr const char *const riscv_assembly_global_var = 
                ".comm %s, 4, 4\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_global_var,
                             p_variable.getNameCString());
        }
        
    }
    else{
        if(kind == SymbolEntry::KindEnum::kConstantKind) {
            // Generate RISC-V instructions for local constant variable
            cur_offset += 4;
            entry->setOffset(cur_offset);
            constexpr const char *const riscv_assembly_local_const_set = 
                "    addi t0, s0, -%d\n"
                "    addi sp, sp, -4\n" 
                "    sw t0, 0(sp)\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_local_const_set, cur_offset);
            p_variable.visitChildNodes(*this);
            constexpr const char *const riscv_assembly_local_const_epilogue = 
                "    lw t0, 0(sp)\n"
                "    addi sp, sp, 4\n"
                "    lw t1, 0(sp)\n"
                 "    addi sp, sp, 4\n"
                 "    sw t0, 0(t1)\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_local_const_epilogue);
        
        }
        else{
            if(kind == SymbolEntry::KindEnum::kVariableKind || kind == SymbolEntry::KindEnum::kParameterKind || kind == SymbolEntry::KindEnum::kLoopVarKind) {
                // Generate RISC-V instructions for local variable
                cur_offset += 4;
                entry->setOffset(cur_offset);
                if(kind == SymbolEntry::KindEnum::kParameterKind) {
                    if(func_param < 8){
                        constexpr const char *const riscv_assembly_local_var = "    sw a%d, -%d(s0)\n"; 
                        dumpInstructions(m_output_file.get(), riscv_assembly_local_var, ((cur_offset - 12) / 4) % 8, cur_offset);
                    }
                    else{
                        constexpr const char *const riscv_assembly_local_var = "    sw t%d, -%d(s0)\n"; 
                        dumpInstructions(m_output_file.get(), riscv_assembly_local_var, func_param % 8 + 2, cur_offset);
                    }
                    func_param++;
                }
            }
        }
    }
}

void CodeGenerator::visit(ConstantValueNode &p_constant_value) {
    
    std::string constant_value = p_constant_value.getConstantValueCString();
    const char* value_const;
    if(constant_value == "true") {
        value_const = "1";
    } else if(constant_value == "false") {
        value_const = "0";
    } else {
        value_const = p_constant_value.getConstantValueCString();
    }

    constexpr const char *const riscv_assembly_constant_value =
        "    li t0, %s\n"
        "    addi sp, sp, -4\n"
        "    sw t0, 0(sp)\n";\
    dumpInstructions(m_output_file.get(), riscv_assembly_constant_value, value_const);
}

void CodeGenerator::visit(FunctionNode &p_function) {
    // Reconstruct the scope for looking up the symbol entry.
    m_symbol_manager.pushScope(
        std::move(m_symbol_table_of_scoping_nodes.at(&p_function)));
    
    constexpr const char *const riscv_assembly_func_header =
        "\n.section    .text\n"
        "    .align 2\n"
        "    .globl %s\n"
        "    .type %s, @function\n\n"
        "%s:\n";

    const char *func_name = p_function.getNameCString();
    dumpInstructions(m_output_file.get(), riscv_assembly_func_header,
                     func_name, func_name, func_name);
                    
    cur_offset = 8;

    constexpr const char *const riscv_assembly_func_prologue =
        "    # in the function prologue\n"
        "    addi sp, sp, -128\n"
        "    sw ra, 124(sp)\n"
        "    sw s0, 120(sp)\n"
        "    addi s0, sp, 128\n\n";
    
    dumpInstructions(m_output_file.get(), riscv_assembly_func_prologue);

    while(func_param > 0){
        if(func_param < 8){
            constexpr const char *const riscv_assembly_func_param = "    sw a%d, -%d(s0)\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_func_param, ((cur_offset - 12) / 4) % 8, cur_offset);
        }
        else{
            constexpr const char *const riscv_assembly_func_param = "    sw t%d, -%d(s0)\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_func_param, func_param % 8 + 2, cur_offset);
        }
        func_param--;
    }
    
    p_function.visitParamChildNodes(*this);
    p_function.visitBodyChildNodes(*this);
    while(func_param > 0){
        cur_offset += 4;
        if(func_param < 8){
            constexpr const char *const riscv_assembly_func_param = "    sw a%d, -%d(s0)\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_func_param, ((cur_offset - 12) / 4) % 8, cur_offset);
        }
        else{
            constexpr const char *const riscv_assembly_func_param = "    sw t%d, -%d(s0)\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_func_param, func_param % 8 + 2, cur_offset);
        }
        func_param--;
    }
    m_symbol_manager.popScope();
    constexpr const char *const riscv_assembly_func_epilogue =
        "    .size %s, .-%s\n";
    dumpInstructions(m_output_file.get(), riscv_assembly_func_epilogue, func_name, func_name);
}

void CodeGenerator::visit(CompoundStatementNode &p_compound_statement) {
    // Reconstruct the scope for looking up the symbol entry.
    m_symbol_manager.pushScope(
        std::move(m_symbol_table_of_scoping_nodes.at(&p_compound_statement)));

    if(if_cnt > 0 || while_cnt > 0) {
        constexpr const char *const riscv_assembly_cnt = 
            "L%d:";
        dumpInstructions(m_output_file.get(), riscv_assembly_cnt, cnt);
        cnt++;
    }
    else if(for_cnt > 0) {
        constexpr const char *const riscv_assembly_for = 
            "    lw t0, 0(sp)\n"
            "    addi sp, sp, 4\n"
            "    lw t1, 0(sp)\n"
            "    addi sp, sp, 4\n"
            "    bge t1, t0, L%d\n"
            "L%d:\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_for, pseudo_cnt+2, pseudo_cnt+1);
    }
    else{
       
        constexpr const char *const riscv_assembly_func_prologue = 
            "    addi sp, sp, -128\n"
            "    sw ra, 124(sp)\n"
            "    sw s0, 120(sp)\n"
            "    addi s0, sp, 128\n\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_func_prologue);
        cur_offset = 8;
        while(func_param > 0){
            cur_offset += 4;
            if(func_param >= 8){
                constexpr const char *const riscv_assembly_func_param = "    sw a%d, -%d(s0)\n";
                dumpInstructions(m_output_file.get(), riscv_assembly_func_param, ((cur_offset - 12) / 4) % 8, cur_offset);
            }
            else{
                constexpr const char *const riscv_assembly_func_param = "    sw t%d, -%d(s0)\n";
                dumpInstructions(m_output_file.get(), riscv_assembly_func_param,  func_param % 8 + 2 ,  cur_offset);
            }
            func_param--;
        }
    }

    p_compound_statement.visitChildNodes(*this);

    if(if_cnt > 0 && else_cnt > 0) {
        dumpInstructions(m_output_file.get(), "    j L%d\n", cnt + 1);
        else_cnt--;
        
    }
    else if(while_cnt > 0){
        dumpInstructions(m_output_file.get(), "    j L%d\n", cnt - 2);
        
    }
    else if(for_cnt > 0 || if_cnt > 0){
        /* Deliberately Void */
    }
    else{
        constexpr const char *const riscv_assembly_func_epilogue = 
            "    lw ra, 124(sp)\n"
            "    lw s0, 120(sp)\n"
            "    addi sp, sp, 128\n"
            "    jr ra\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_func_epilogue);
    }

    m_symbol_manager.popScope();
}

void CodeGenerator::visit(PrintNode &p_print) {
    constexpr const char *const riscv_assembly_print = 
        "    lw a0, 0(sp)\n"
        "    addi sp, sp, 4\n"
        "    jal ra, printInt\n";
    p_print.visitChildNodes(*this);
    dumpInstructions(m_output_file.get(), riscv_assembly_print);
}

void CodeGenerator::visit(BinaryOperatorNode &p_bin_op) {
    op_cnt++;
    p_bin_op.visitChildNodes(*this);
    op_cnt--;

    constexpr const char *const riscv_assembly_bin_op_prologue = 
        "    lw t0, 0(sp)\n"
        "    addi sp, sp, 4\n"
        "    lw t1, 0(sp)\n"
        "    addi sp, sp, 4\n";
    
    constexpr const char *const riscv_assembly_binop_epilogue = 
        "    addi sp, sp, -4\n"
        "    sw t0, 0(sp)\n";
    dumpInstructions(m_output_file.get(), riscv_assembly_bin_op_prologue);
    
    Operator op = p_bin_op.getOp();
    if(op == Operator::kMultiplyOp){
        constexpr const char* const riscv_assembly_multiply_op = 
            "    mul t0, t1, t0\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_multiply_op);
        dumpInstructions(m_output_file.get(), riscv_assembly_binop_epilogue);
    }
    else if(op == Operator::kDivideOp){
        constexpr const char* const riscv_assembly_divide_op = 
            "    div t0, t1, t0\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_divide_op);
        dumpInstructions(m_output_file.get(), riscv_assembly_binop_epilogue);
    }
    else if(op == Operator::kModOp){
        constexpr const char* const riscv_assembly_mod_op = 
            "    rem t0, t1, t0\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_mod_op);
        dumpInstructions(m_output_file.get(), riscv_assembly_binop_epilogue);
    }
    else if(op == Operator::kPlusOp){
        constexpr const char* const riscv_assembly_plus_op = 
            "    add t0, t1, t0\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_plus_op);
        dumpInstructions(m_output_file.get(), riscv_assembly_binop_epilogue);
    }
    else if(op == Operator::kMinusOp){
        constexpr const char* const riscv_assembly_minus_op = 
            "    sub t0, t1, t0\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_minus_op);
        dumpInstructions(m_output_file.get(), riscv_assembly_binop_epilogue);
    }
    else if(op == Operator::kLessOp){
        constexpr const char* const riscv_assembly_less_op = 
            "    bge t1, t0, L%d\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_less_op, cnt + 1);
    }
    else if(op == Operator::kLessOrEqualOp){
        constexpr const char* const riscv_assembly_less_or_equal_op = 
            "    bgt t1, t0, L%d\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_less_or_equal_op, cnt + 1);
    }
    else if(op == Operator::kGreaterOp){
        constexpr const char* const riscv_assembly_greater_op = 
            "    ble t1, t0, L%d\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_greater_op, cnt + 1);
    }
    else if(op == Operator::kGreaterOrEqualOp){
        constexpr const char* const riscv_assembly_greater_or_equal_op = 
            "    blt t1, t0, L%d\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_greater_or_equal_op, cnt + 1);
    }
    else if(op == Operator::kEqualOp){
        constexpr const char* const riscv_assembly_equal_op = 
            "    bne t1, t0, L%d\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_equal_op, cnt + 1);
    }
    else if(op == Operator::kNotEqualOp){
        constexpr const char* const riscv_assembly_not_equal_op = 
            "    beq t1, t0, L%d\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_not_equal_op, cnt + 1);
    }
    else if(op == Operator::kAndOp){
        constexpr const char* const riscv_assembly_and_op = 
            "    and t0, t1, t0\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_and_op);
        dumpInstructions(m_output_file.get(), riscv_assembly_binop_epilogue);
    }
    else if(op == Operator::kOrOp){
        constexpr const char* const riscv_assembly_or_op = 
            "    or t0, t1, t0\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_or_op);
        dumpInstructions(m_output_file.get(), riscv_assembly_binop_epilogue);
    }

}

void CodeGenerator::visit(UnaryOperatorNode &p_un_op) {
    p_un_op.visitChildNodes(*this);

    constexpr const char *const riscv_assembly_unary_op_prologue = 
        "    lw t0, 0(sp)\n"
        "    addi sp, sp, 4\n";
    
    constexpr const char *const riscv_assembly_unary_op_epilogue = 
        "    addi sp, sp, -4\n"
        "    sw t0, 0(sp)\n";
    
    dumpInstructions(m_output_file.get(), riscv_assembly_unary_op_prologue);
    
    Operator op = p_un_op.getOp();
    if(op == Operator::kNegOp) {
        constexpr const char *const riscv_assembly_neg_op = 
            "    neg t0, t0\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_neg_op);
        dumpInstructions(m_output_file.get(), riscv_assembly_unary_op_epilogue);
    }
    else if(op == Operator::kNotOp) {
        constexpr const char *const riscv_assembly_not_op = 
            "    li t1, -1\n"
            "    add t0, t0, t1\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_not_op);
        dumpInstructions(m_output_file.get(), riscv_assembly_unary_op_epilogue);
    }
    
}

void CodeGenerator::visit(FunctionInvocationNode &p_func_invocation) {
    func_invocation_cnt++;
    p_func_invocation.visitChildNodes(*this);
    func_invocation_cnt--;

    for(int i = 0; i < p_func_invocation.getArguments().size(); i++) {
        if(i < 8) {
            constexpr const char *const riscv_assembly_func_arg = 
                "    lw a%d, 0(sp)\n"
                "    addi sp, sp, 4\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_func_arg, i);
        }
        else{
            constexpr const char *const riscv_assembly_func_arg = 
                "    lw t%d, 0(sp)\n"
                "    addi sp, sp, 4\n";  
            dumpInstructions(m_output_file.get(), riscv_assembly_func_arg, i % 8 + 2);
        }
    }

    constexpr const char *const riscv_assembly_func_invocation_epilogue = 
        "    jal ra, %s\n"
        "    mv t0, a0\n"
        "    addi sp, sp, -4\n"
        "    sw t0, 0(sp)\n";   

    dumpInstructions(m_output_file.get(), riscv_assembly_func_invocation_epilogue, p_func_invocation.getNameCString());
}

void CodeGenerator::visit(VariableReferenceNode &p_variable_ref) {
    SymbolEntry *entry = m_symbol_manager.lookup(p_variable_ref.getName());
    bool isGlobal = (entry->getLevel() == 0);
    int cur_offset_ici = entry->getOffset();
    SymbolEntry::KindEnum kind = entry->getKind();

    if(kind == SymbolEntry::KindEnum::kParameterKind){
        constexpr const char *const riscv_assembly_local = 
            "    lw t0, -%d(s0)\n"
            "    addi sp, sp, -4\n"
            "    sw t0, 0(sp)\n";
        dumpInstructions(m_output_file.get(), riscv_assembly_local, cur_offset_ici);
    }
    else if(isGlobal){
        if((isAssignmentLeft || isRead) && op_cnt == 0 && func_invocation_cnt == 0) {
            constexpr const char *const riscv_assembly_global = 
                "    la t0, %s\n"
                "    addi sp, sp, -4\n"
                "    sw t0, 0(sp)\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_global, p_variable_ref.getNameCString());
        }
        else{
            constexpr const char *const riscv_assembly_global = 
                "    la t0, %s\n"
                "    lw t1, 0(t0)\n"
                "    mv t0, t1\n"
                "    addi sp, sp, -4\n"
                "    sw t0, 0(sp)\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_global, p_variable_ref.getNameCString());
        }
    }
    else{
        if((isAssignmentLeft || isRead) && op_cnt == 0 && func_invocation_cnt == 0) {
            constexpr const char *const riscv_assembly_local = 
                "    addi t0, s0, -%d\n"
                "    addi sp, sp, -4\n"
                "    sw t0, 0(sp)\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_local, cur_offset_ici);
        }
        else{
            constexpr const char *const riscv_assembly_local = 
                "    lw t0, -%d(s0)\n"
                "    addi sp, sp, -4\n"
                "    sw t0, 0(sp)\n";
            dumpInstructions(m_output_file.get(), riscv_assembly_local, cur_offset_ici);
        }
    }

}

void CodeGenerator::visit(AssignmentNode &p_assignment) {
    isAssignmentLeft = true;
    p_assignment.visitChildNodes(*this);
    isAssignmentLeft = false;

    constexpr const char *const riscv_assembly_assignment =
        "    lw t0, 0(sp)\n"
        "    addi sp, sp, 4\n"
        "    lw t1, 0(sp)\n"
        "    addi sp, sp, 4\n"
        "    sw t0, 0(t1)\n";

    dumpInstructions(m_output_file.get(), riscv_assembly_assignment);

    if(for_assignment_cnt > 0){
        constexpr const char *const riscv_assembly_for_assignment =
            "L%d:\n"
            "    lw t0, -%d(s0)\n"
            "    addi sp, sp, -4\n"
            "    sw t0, 0(sp)\n";

        dumpInstructions(m_output_file.get(), riscv_assembly_for_assignment, pseudo_cnt, cur_offset);
        for_assignment_cnt--;
    }
}

void CodeGenerator::visit(ReadNode &p_read) {
    isRead = true;
    p_read.visitChildNodes(*this);
    isRead = false;

    constexpr const char *const riscv_assembly_read =
        "    jal ra, readInt\n"
        "    lw t0, 0(sp)\n"
        "    addi sp, sp, 4\n"
        "    sw a0, 0(t0)\n";

    dumpInstructions(m_output_file.get(), riscv_assembly_read);
}

void CodeGenerator::visit(IfNode &p_if) {
    if_cnt++;
    if(p_if.ifElseBodyExists()) {
        else_cnt++;
    }
    p_if.visitChildNodes(*this);
    if_cnt--;

    dumpInstructions(m_output_file.get(), "L%d:", cnt);
    cnt++;
}

void CodeGenerator::visit(WhileNode &p_while) {
    dumpInstructions(m_output_file.get(), "\nL%d:\n", cnt);
    cnt++;
    while_cnt++;
    p_while.visitChildNodes(*this);
    while_cnt--;
    dumpInstructions(m_output_file.get(), "\nL%d:\n", cnt);
    cnt++;

}

void CodeGenerator::visit(ForNode &p_for) {
    // Reconstruct the scope for looking up the symbol entry.
    m_symbol_manager.pushScope(
        std::move(m_symbol_table_of_scoping_nodes.at(&p_for)));

    constexpr const char *const riscv_assembly_for_n =
        "    addi t0, s0, -%d\n"
        "    addi sp, sp, -4\n"
        "    sw t0, 0(sp)\n"
        "    lw t0, -%d(s0)\n"
        "    addi sp, sp, -4\n"
        "    sw t0, 0(sp)\n"
        "    li t0, 1\n"
        "    addi sp, sp, -4\n"
        "    sw t0, 0(sp)\n"
        "    lw t0, 0(sp)\n"
        "    addi sp, sp, 4\n"
        "    lw t1, 0(sp)\n"
        "    addi sp, sp, 4\n"
        "    add t0, t1, t0\n"
        "    addi sp, sp, -4\n"
        "    sw t0, 0(sp)\n"
        "    lw t0, 0(sp)\n"
        "    addi sp, sp, 4\n"
        "    lw t1, 0(sp)\n"
        "    addi sp, sp, 4\n"
        "    sw t0, 0(t1)\n"
        "    j L%d\n"
        "L%d:\n";

    pseudo_cnt = cnt;
    cnt_loop.push_back(cnt);
    cnt += 3;
    for_cnt++;
    for_assignment_cnt++;
    p_for.visitChildNodes(*this);
    
    for_cnt--;
    dumpInstructions(m_output_file.get(), riscv_assembly_for_n, cur_offset, cur_offset, pseudo_cnt, pseudo_cnt + 2);

    // Remove the entries in the hash table
    m_symbol_manager.popScope();

    cur_offset -= 4;
    cnt_loop.pop_back();
    if(cnt_loop.size()) {
        pseudo_cnt = cnt_loop.back();
    } else {
        pseudo_cnt = 0;
    }
}

void CodeGenerator::visit(ReturnNode &p_return) {
    p_return.visitChildNodes(*this);

    constexpr const char *const riscv_assembly_return =
        "    lw a0, 0(sp)\n"
        "    addi sp, sp, 4\n"
        "    mv a0, t0\n"
        "\n    # in the function epilogue\n"
        "    lw ra, 124(sp)\n"
        "    lw s0, 120(sp)\n"
        "    addi sp, sp, 128\n"
        "    jr ra\n";

    dumpInstructions(m_output_file.get(), riscv_assembly_return);
}
