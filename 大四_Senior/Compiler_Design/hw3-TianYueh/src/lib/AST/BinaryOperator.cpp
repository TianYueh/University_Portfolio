#include "AST/BinaryOperator.hpp"

// TODO
BinaryOperatorNode::BinaryOperatorNode(const uint32_t line, const uint32_t col,
                                       std::string p_operator,
                                       ExpressionNode *const p_left_expression,
                                       ExpressionNode *const p_right_expression)
    : ExpressionNode{line, col}, m_operator(p_operator),
      m_left_expression(p_left_expression),
      m_right_expression(p_right_expression) {}

// TODO: You may use code snippets in AstDumper.cpp
void BinaryOperatorNode::print() {}

const char *BinaryOperatorNode::getOperatorCString() {
    return m_operator.c_str();
}

void BinaryOperatorNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    // TODO
    if (m_left_expression != nullptr) {
        m_left_expression->accept(p_visitor);
    }
    if (m_right_expression != nullptr) {
        m_right_expression->accept(p_visitor);
    }
    
}
