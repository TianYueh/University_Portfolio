#ifndef __AST_IF_NODE_H
#define __AST_IF_NODE_H

#include "AST/ast.hpp"

class IfNode : public AstNode {
  public:
    IfNode(const uint32_t line, const uint32_t col,
            AstNode* p_expr, AstNode* c1, AstNode* c2
           /* TODO: expression, compound statement, compound statement */);
    ~IfNode() = default;

    void print() override;
    void accept(AstNodeVisitor &p_visitor) override;
    void visitChildNodes(AstNodeVisitor &p_visitor);

  private:
    // TODO: expression, compound statement, compound statement
    AstNode* expr;
    AstNode* c1;
    AstNode* c2;
};

#endif
