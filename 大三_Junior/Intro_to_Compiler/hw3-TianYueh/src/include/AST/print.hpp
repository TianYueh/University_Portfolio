#ifndef __AST_PRINT_NODE_H
#define __AST_PRINT_NODE_H

#include "AST/ast.hpp"

class PrintNode : public AstNode {
  public:
    PrintNode(const uint32_t line, const uint32_t col,
              AstNode* p_expression
              /* TODO: expression */);
    ~PrintNode() = default;

    void print() override;
    void accept(AstNodeVisitor &p_node_visitor) override;
    void visitChildNodes(AstNodeVisitor &p_visitor);

  private:
    // TODO: expression
    AstNode *expression;
};

#endif
