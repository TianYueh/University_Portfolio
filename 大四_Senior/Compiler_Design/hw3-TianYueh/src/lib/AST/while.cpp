#include "AST/while.hpp"

// TODO
WhileNode::WhileNode(const uint32_t line, const uint32_t col,
                     ExpressionNode *p_expression,
                     CompoundStatementNode *p_compound_statement)
    : AstNode{line, col}, 
      m_expression(p_expression),
      m_compound_statement(p_compound_statement) {}

// TODO: You may use code snippets in AstDumper.cpp
void WhileNode::print() {}

void WhileNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    // TODO
    if (m_expression != nullptr) {
        m_expression->accept(p_visitor);
    }
    if (m_compound_statement != nullptr) {
        m_compound_statement->accept(p_visitor);
    }
}
