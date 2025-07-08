#include "AST/ConstantValue.hpp"

// TODO
ConstantValueNode::ConstantValueNode(const uint32_t line, const uint32_t col, 
                                     const char* p_constant_value)
    : ExpressionNode{line, col}, m_constant_value(p_constant_value) {}

// TODO: You may use code snippets in AstDumper.cpp
void ConstantValueNode::print() {}

const char* ConstantValueNode::getConstantValue() {
    return m_constant_value.c_str();
}
