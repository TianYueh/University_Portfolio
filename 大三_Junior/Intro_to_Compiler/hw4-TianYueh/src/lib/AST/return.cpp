#include "AST/return.hpp"

void ReturnNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    m_ret_val->accept(p_visitor);
}

const int ReturnNode::checkInvalidRetType() const {
    if ((std::string)m_ret_val->getPTypeCString() == "null") {
        return 1;
    }
    return 0;
}

const char* ReturnNode::getRetTypeCString() const {
    return m_ret_val->getPTypeCString();
}

const uint32_t ReturnNode::getRetLocationCol() const {
    return m_ret_val->getLocation().col;
}
