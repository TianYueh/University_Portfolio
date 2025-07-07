#include "AST/read.hpp"

void ReadNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    m_target->accept(p_visitor);
}

const char* ReadNode::getTargetNameCString() const {
    return m_target->getNameCString();
}

const int ReadNode::checkTargetScalarType() const {
    std::string type = m_target->getPTypeCString();
    if (type == "integer" || type == "real" || type == "boolean" || type == "string") {
        return 1;
    }
    return 0;
}

const int ReadNode::checkInvalidChildren() const {
    if ((std::string)m_target->getPTypeCString() == "null") {
        return 1;
    }
    return 0;
}

const uint32_t ReadNode::getTargetLocationCol() const {
    return m_target->getLocation().col;
}
