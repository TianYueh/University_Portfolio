#include "AST/UnaryOperator.hpp"

void UnaryOperatorNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    auto visit_ast_node = [&](auto &ast_node) { ast_node->accept(p_visitor); };

    visit_ast_node(m_operand);
}

const char* UnaryOperatorNode::getOperandTypeCString() const {
    return m_operand->getPTypeCString();
}

const int UnaryOperatorNode::checkInvalidChildren() const {
    if ((std::string)m_operand->getPTypeCString() == "null") {
        return 1;
    }
    return 0;
}
