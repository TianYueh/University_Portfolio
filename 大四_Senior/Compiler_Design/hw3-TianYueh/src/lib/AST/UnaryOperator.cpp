#include "AST/UnaryOperator.hpp"

// TODO
UnaryOperatorNode::UnaryOperatorNode(const uint32_t line, const uint32_t col,
                                     std::string op,
                                     ExpressionNode *const p_expression)
    : ExpressionNode{line, col},
      m_op(op),
      m_expression(p_expression) {}

// TODO: You may use code snippets in AstDumper.cpp
void UnaryOperatorNode::print() {}

const char* UnaryOperatorNode::getOperatorCString() {
    return m_op.c_str();
}

void UnaryOperatorNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    // TODO
    if (m_expression != nullptr) {
        m_expression->accept(p_visitor);
    }
}
