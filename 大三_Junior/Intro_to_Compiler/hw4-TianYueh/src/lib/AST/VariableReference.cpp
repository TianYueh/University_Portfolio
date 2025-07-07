#include "AST/VariableReference.hpp"

#include <algorithm>
#include <iostream>

#include <string.h>

void VariableReferenceNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    auto visit_ast_node = [&](auto &ast_node) { ast_node->accept(p_visitor); };

    for_each(m_indices.begin(), m_indices.end(), visit_ast_node);
}

int VariableReferenceNode::checkInvalidChildren() {
    for (auto &index : m_indices) {
        if (strcmp(index->getPTypeCString(), "null") == 0) {
            return 1;
        }
    }
    return 0;
}

int VariableReferenceNode::getIndicesNum() const {
    return m_indices.size();
}

int VariableReferenceNode::checkNonIntegerIndex(){
    for (auto &index : m_indices) {
        if (strcmp(index->getPTypeCString(), "integer") != 0) {
            return index->getLocation().col;
        }
    }
    return -1;
}