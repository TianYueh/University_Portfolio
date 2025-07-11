#include "AST/read.hpp"

// TODO
ReadNode::ReadNode(const uint32_t line, const uint32_t col,
                    AstNode* p_var_ref)
    : AstNode{line, col}, var_ref(p_var_ref) {}

// TODO: You may use code snippets in AstDumper.cpp
void ReadNode::print() {}

void ReadNode::accept(AstNodeVisitor &p_visitor){
    p_visitor.visit(*this);
}

void ReadNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    if(var_ref != nullptr){
        var_ref->accept(p_visitor);
    }
}
