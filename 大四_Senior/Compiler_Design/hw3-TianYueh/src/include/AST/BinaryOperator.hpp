#ifndef __AST_BINARY_OPERATOR_NODE_H
#define __AST_BINARY_OPERATOR_NODE_H

#include "AST/expression.hpp"
#include "visitor/AstNodeVisitor.hpp"

#include <memory>
#include <string>
class BinaryOperatorNode : public ExpressionNode {
  public:
    BinaryOperatorNode(const uint32_t line, const uint32_t col,
                       /* TODO: operator, expressions */
                      std::string p_operator,
                      ExpressionNode *const p_left_expression,
                      ExpressionNode *const p_right_expression
                      );
                    ;
    ~BinaryOperatorNode() = default;

    void print() override;
    void accept(AstNodeVisitor &p_visitor) override {
        p_visitor.visit(*this);
    }
    void visitChildNodes(AstNodeVisitor &p_visitor) override;
    const char *getOperatorCString();

  private:
    // TODO: operator, expressions
    std::string m_operator;
    ExpressionNode *m_left_expression;
    ExpressionNode *m_right_expression;
};

#endif
