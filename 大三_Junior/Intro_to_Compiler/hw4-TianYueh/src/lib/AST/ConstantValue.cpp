#include "AST/ConstantValue.hpp"

const PTypeSharedPtr& ConstantValueNode::getTypeSharedPtr() const {
    return m_constant_ptr->getTypeSharedPtr();
}

const char* ConstantValueNode::getConstantValueCString() const {
    return m_constant_ptr->getConstantValueCString();
}