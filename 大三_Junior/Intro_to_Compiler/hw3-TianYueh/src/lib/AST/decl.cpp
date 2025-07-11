#include "AST/decl.hpp"

// TODO
DeclNode::DeclNode(const uint32_t line, const uint32_t col, 
                    std::vector<AstNode*> *p_var_list)
    : AstNode{line, col}, var_list(p_var_list){}

// TODO
//DeclNode::DeclNode(const uint32_t line, const uint32_t col)
//    : AstNode{line, col} {}

// TODO: You may use code snippets in AstDumper.cpp
void DeclNode::print() {

}

void DeclNode::accept(AstNodeVisitor &p_visitor){
    p_visitor.visit(*this);
}

void DeclNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    if(var_list != nullptr){
        for(auto &var_list : *var_list){
            var_list->accept(p_visitor);
        }
    }
}

std::vector<AstNode*>* DeclNode::getVarList(){
    return var_list;
}
