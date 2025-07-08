#include "AST/variable.hpp"

// TODO
VariableNode::VariableNode(const uint32_t line, const uint32_t col,
                           const char* p_variable_name,
                           const char* p_variable_type,
                           ConstantValueNode *p_constant_value)
    : AstNode{line, col}, 
      m_variable_name(p_variable_name),
      m_variable_type(p_variable_type),
      m_constant_value(p_constant_value) {}

// TODO: You may use code snippets in AstDumper.cpp
void VariableNode::print() {}

void VariableNode::visitChildNodes(AstNodeVisitor &p_visitor) {
     // TODO
    if (m_constant_value != nullptr) {
        m_constant_value->accept(p_visitor);
    }

}
