#include "AST/for.hpp"

// TODO
ForNode::ForNode(const uint32_t line, const uint32_t col,
                 DeclNode *const p_declaration,
                 AssignmentNode *const p_assignment,
                 ExpressionNode *const p_expression,
                 CompoundStatementNode *const p_compound_statement)
    : AstNode{line, col}, m_declaration{p_declaration},
      m_assignment{p_assignment}, m_expression{p_expression},
      m_compound_statement{p_compound_statement} {}

// TODO: You may use code snippets in AstDumper.cpp
void ForNode::print() {}

void ForNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    // TODO
    if (m_declaration != nullptr) {
        m_declaration->accept(p_visitor);
    }
    if (m_assignment != nullptr) {
        m_assignment->accept(p_visitor);
    }
    if (m_expression != nullptr) {
        m_expression->accept(p_visitor);
    }
    if (m_compound_statement != nullptr) {
        m_compound_statement->accept(p_visitor);
    }

}
