#include "AST/VariableReference.hpp"

// TODO
VariableReferenceNode::VariableReferenceNode(const uint32_t line,
                                             const uint32_t col,
                                             std::string p_variable_name)
    : ExpressionNode{line, col}, m_variable_name(p_variable_name) {}

// TODO
VariableReferenceNode::VariableReferenceNode(const uint32_t line,
                                              const uint32_t col,
                                              std::string p_variable_name,
                                              std::vector<ExpressionNode*> *p_expressions)
    : ExpressionNode{line, col}, m_variable_name(p_variable_name), m_expressions(p_expressions) {}

// TODO: You may use code snippets in AstDumper.cpp
void VariableReferenceNode::print() {}

const char* VariableReferenceNode::getVariableName() {
    return m_variable_name.c_str();
}

void VariableReferenceNode::visitChildNodes(AstNodeVisitor &p_visitor) {
     // TODO
    if (m_expressions != nullptr) {
        for (auto expression : *m_expressions) {
            expression->accept(p_visitor);
        }
    }
}
