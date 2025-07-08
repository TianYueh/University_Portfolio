#include "AST/assignment.hpp"

// TODO
AssignmentNode::AssignmentNode(const uint32_t line, const uint32_t col,
                               VariableReferenceNode *const p_variable_ref,
                               ExpressionNode *const p_expression)
    : AstNode{line, col}, m_variable_ref{p_variable_ref}, m_expression{p_expression} {}

// TODO: You may use code snippets in AstDumper.cpp
void AssignmentNode::print() {}

void AssignmentNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    if (m_variable_ref != nullptr) {
        m_variable_ref->accept(p_visitor);
    }
    if (m_expression != nullptr) {
        m_expression->accept(p_visitor);
    }
}
