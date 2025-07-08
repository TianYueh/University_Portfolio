#ifndef __AST_IF_NODE_H
#define __AST_IF_NODE_H

#include "AST/ast.hpp"
#include "AST/expression.hpp"
#include "AST/CompoundStatement.hpp"
#include "visitor/AstNodeVisitor.hpp"
#include <string>
#include <vector>

class IfNode : public AstNode {
  public:
    IfNode(const uint32_t line, const uint32_t col,
           /* TODO: expression, compound statement, compound statement */
           ExpressionNode *p_expression,
           CompoundStatementNode *p_compound_statement,
           CompoundStatementNode *p_else_compound_statement);
    ~IfNode() = default;

    void print() override;
    void accept(AstNodeVisitor &p_visitor) override {
        p_visitor.visit(*this);
    }
    void visitChildNodes(AstNodeVisitor &p_visitor) override;
    

  private:
    // TODO: expression, compound statement, compound statement
    ExpressionNode *m_expression;
    CompoundStatementNode *m_compound_statement;
    CompoundStatementNode *m_else_compound_statement;
};

#endif
