#include "AST/decl.hpp"

// TODO
DeclNode::DeclNode(const uint32_t line, const uint32_t col, std::vector<VariableNode*> *const p_var_list)
    : AstNode{line, col}, m_var_list(*p_var_list) {}

// TODO
//DeclNode::DeclNode(const uint32_t line, const uint32_t col)
//    : AstNode{line, col} {}

// TODO: You may use code snippets in AstDumper.cpp
void DeclNode::print() {}


void DeclNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    // TODO
    if (m_var_list.size() > 0) {
        for (auto var : m_var_list) {
            var->accept(p_visitor);
        }
    }
}

std::vector<const char*> DeclNode::getDeclType() {
    std::vector<const char*> decltypes;
    for (auto &var : m_var_list) {
        const char* type = var->getVariableType();
        decltypes.push_back(type);
    }
    return decltypes;
}
