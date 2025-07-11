#include "AST/print.hpp"

void PrintNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    m_target->accept(p_visitor);
}

const int PrintNode::checkInvalidChildren() const {
    if ((std::string)m_target->getPTypeCString() == "null") {
        return 1;
    }
    return 0;
}

const uint32_t PrintNode::getTargetLocationCol() const {
    return m_target->getLocation().col;
}

const int PrintNode::checkTargetScalarType() const {
    std::string type = m_target->getPTypeCString();
    if (type == "integer" || type == "real" || type == "boolean" || type == "string") {
        return 1;
    }
    else{
        return 0;
    }
}