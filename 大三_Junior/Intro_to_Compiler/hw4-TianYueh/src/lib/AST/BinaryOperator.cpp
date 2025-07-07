#include "AST/BinaryOperator.hpp"

#include <string.h>

void BinaryOperatorNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    auto visit_ast_node = [&](auto &ast_node) { ast_node->accept(p_visitor); };

    visit_ast_node(m_left_operand);
    visit_ast_node(m_right_operand);
}

const char* BinaryOperatorNode::getLTypeCString() const {
    return m_left_operand->getPTypeCString();
}

const char* BinaryOperatorNode::getRTypeCString() const {
    return m_right_operand->getPTypeCString();
}

const int BinaryOperatorNode::checkInvalidChildren() const{
    if(m_left_operand->getPTypeCString() == "null"){
        return 1;
    }
        
    if(strcmp(m_right_operand->getPTypeCString(), "null") == 0){
        return 1;
    }
    return 0;
}