#include "AST/if.hpp"

// TODO
IfNode::IfNode(const uint32_t line, const uint32_t col,
                ExpressionNode *p_expression,
                CompoundStatementNode *p_compound_statement,
                CompoundStatementNode *p_else_compound_statement
                )
    : AstNode{line, col}, 
      m_expression(p_expression),
      m_compound_statement(p_compound_statement),
      m_else_compound_statement(p_else_compound_statement) {}

// TODO: You may use code snippets in AstDumper.cpp
void IfNode::print() {}

void IfNode::visitChildNodes(AstNodeVisitor &p_visitor) {
//     // TODO
    if (m_expression != nullptr) {
        m_expression->accept(p_visitor);
    }
    if (m_compound_statement != nullptr) {
        m_compound_statement->accept(p_visitor);
    }
    if (m_else_compound_statement != nullptr) {
        m_else_compound_statement->accept(p_visitor);
    }
}
