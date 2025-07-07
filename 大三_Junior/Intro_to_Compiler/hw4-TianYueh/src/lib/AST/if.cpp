#include "AST/if.hpp"

void IfNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    m_condition->accept(p_visitor);  
    m_body->accept(p_visitor);
    if (m_else_body) {
        m_else_body->accept(p_visitor);
    }
}

const int IfNode::checkInvalidCondition() const {
    if ((std::string)m_condition->getPTypeCString() == "null") {
        return 1;
    }
    return 0;
}

const int IfNode::checkConditionBoolType() const{
    if ((std::string)m_condition->getPTypeCString() == "boolean") {
        return 1;
    }
    return 0;
}

const uint32_t IfNode::getConditionLocationCol() const {
    return m_condition->getLocation().col;
}