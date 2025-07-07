#include "AST/for.hpp"

// TODO
ForNode::ForNode(const uint32_t line, const uint32_t col,
                AstNode* p_decl, AstNode* p_assign, AstNode* p_expr, AstNode* p_compound)
    : AstNode{line, col}, decl(p_decl), assign(p_assign), expr(p_expr), compound(p_compound){}

// TODO: You may use code snippets in AstDumper.cpp
void ForNode::print() {}

void ForNode::accept(AstNodeVisitor &p_visitor){
    p_visitor.visit(*this);
}

void ForNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    if(decl != nullptr){
        decl->accept(p_visitor);
    }
    if(assign != nullptr){
        assign->accept(p_visitor);
    }
    if(expr != nullptr){
        expr->accept(p_visitor);
    }
    if(compound != nullptr){
        compound->accept(p_visitor);
    }
}
