#ifndef AST_AST_DUMPER_H
#define AST_AST_DUMPER_H

#include "util/Indenter.hpp"
#include "visitor/AstNodeVisitor.hpp"

#include <cstdint>

class AstDumper final : public AstNodeVisitor {
  private:
    Indenter m_indenter{' ', 2};

  public:
    ~AstDumper() = default;
    AstDumper() = default;

    void visit(ProgramNode &p_program) override;
    void visit(DeclNode &p_decl) override;
    void visit(VariableNode &p_variable) override;
    void visit(ConstantValueNode &p_constant_value) override;
    void visit(FunctionNode &p_function) override;
    void visit(CompoundStatementNode &p_compound_statement) override;
    void visit(PrintNode &p_print) override;
    void visit(BinaryOperatorNode &p_bin_op) override;
    void visit(UnaryOperatorNode &p_un_op) override;
    void visit(FunctionInvocationNode &p_func_invocation) override;
    void visit(VariableReferenceNode &p_variable_ref) override;
    void visit(AssignmentNode &p_assignment) override;
    void visit(ReadNode &p_read) override;
    void visit(IfNode &p_if) override;
    void visit(WhileNode &p_while) override;
    void visit(ForNode &p_for) override;
    void visit(ReturnNode &p_return) override;

  private:
    void printIndent() const;
    
};

#endif
