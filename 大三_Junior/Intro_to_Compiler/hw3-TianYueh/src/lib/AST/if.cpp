#include "AST/if.hpp"

// TODO
IfNode::IfNode(const uint32_t line, const uint32_t col,
                AstNode* p_expr, AstNode* p_c1, AstNode* p_c2)
    : AstNode{line, col}, expr(p_expr), c1(p_c1), c2(p_c2){}

// TODO: You may use code snippets in AstDumper.cpp
void IfNode::print() {}

void IfNode::accept(AstNodeVisitor &p_visitor){
    p_visitor.visit(*this);
}

void IfNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    if(expr != nullptr){
        expr->accept(p_visitor);
    }
    if(c1 != nullptr){
        c1->accept(p_visitor);
    }
    if(c2 != nullptr){
        c2->accept(p_visitor);
    }
}
