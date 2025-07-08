#include "AST/FunctionInvocation.hpp"

// TODO
FunctionInvocationNode::FunctionInvocationNode(const uint32_t line,
                                               const uint32_t col,
                                               std::string function_name,
                                               std::vector<ExpressionNode *> *p_expressions)
    : ExpressionNode{line, col}, 
      m_function_name(function_name),
      m_expressions(p_expressions) {}

// TODO: You may use code snippets in AstDumper.cpp
void FunctionInvocationNode::print() {}

const char *FunctionInvocationNode::getNameCString() const {
    return m_function_name.c_str();
}

void FunctionInvocationNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    // TODO
    if (m_expressions != nullptr) {
        for (auto expr : *m_expressions) {
            expr->accept(p_visitor);
        }
    }
}
