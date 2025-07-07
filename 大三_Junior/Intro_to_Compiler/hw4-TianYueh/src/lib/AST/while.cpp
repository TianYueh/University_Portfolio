#include "AST/while.hpp"

void WhileNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    m_condition->accept(p_visitor);
    m_body->accept(p_visitor);
}

const int WhileNode::checkInvalidCondition() const {
    if ((std::string)m_condition->getPTypeCString() == "null") {
        return 1;
    }
    return 0;
}

const int WhileNode::checkConditionBoolType() const{
    if ((std::string)m_condition->getPTypeCString() == "boolean") {
        return 1;
    }
    return 0;
}
