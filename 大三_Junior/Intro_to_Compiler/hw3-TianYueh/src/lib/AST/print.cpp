#include "AST/print.hpp"

// TODO
PrintNode::PrintNode(const uint32_t line, const uint32_t col,
                    AstNode* p_expression)
    : AstNode{line, col}, expression(p_expression) {}

// TODO: You may use code snippets in AstDumper.cpp
void PrintNode::print() {}

void PrintNode::accept(AstNodeVisitor &p_visitor){
    p_visitor.visit(*this);
}

void PrintNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    if(expression != nullptr){
        expression->accept(p_visitor);
    }
}
