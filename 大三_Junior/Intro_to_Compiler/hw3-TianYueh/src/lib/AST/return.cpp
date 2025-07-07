#include "AST/return.hpp"

// TODO
ReturnNode::ReturnNode(const uint32_t line, const uint32_t col,
                        AstNode* p_rtnval)
    : AstNode{line, col}, rtnval(p_rtnval) {}

// TODO: You may use code snippets in AstDumper.cpp
void ReturnNode::print() {}

void ReturnNode::accept(AstNodeVisitor &p_visitor){
    p_visitor.visit(*this);
}

void ReturnNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    if(rtnval != nullptr){
        rtnval->accept(p_visitor);
    }
}
