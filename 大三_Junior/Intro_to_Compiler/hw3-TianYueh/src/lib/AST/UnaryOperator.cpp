#include "AST/UnaryOperator.hpp"

// TODO
UnaryOperatorNode::UnaryOperatorNode(const uint32_t line, const uint32_t col,
                                    Unary_Op p_op, AstNode* p_expr)
    : ExpressionNode{line, col}, op(p_op), expr(p_expr) {}

// TODO: You may use code snippets in AstDumper.cpp
void UnaryOperatorNode::print() {}

void UnaryOperatorNode::accept(AstNodeVisitor &p_visitor){
    p_visitor.visit(*this);
}

void UnaryOperatorNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    if(expr != nullptr){
        expr->accept(p_visitor);
    }
}

const char* UnaryOperatorNode::getOperatorCString() const{
    if(op.neg){
        return "neg";
    }
    else if(op.NOT){
        return "not";
    }
}
