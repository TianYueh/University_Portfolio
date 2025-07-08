#ifndef __AST_VARIABLE_REFERENCE_NODE_H
#define __AST_VARIABLE_REFERENCE_NODE_H

#include "AST/expression.hpp"
#include "visitor/AstNodeVisitor.hpp"
#include <string>
#include <vector>

class VariableReferenceNode : public ExpressionNode {
  public:
    // normal reference
    VariableReferenceNode(const uint32_t line, const uint32_t col,
    //                       /* TODO: name */
                          std::string p_variable_name
                          
    );    
    // array reference
    VariableReferenceNode(const uint32_t line, const uint32_t col,
                           /* TODO: name, expressions */
                          std::string p_variable_name,
                          
                          std::vector<ExpressionNode*> *p_expressions);
    ~VariableReferenceNode() = default;

    void print() override;
    void accept(AstNodeVisitor &p_visitor) override {
        p_visitor.visit(*this);
    }
    void visitChildNodes(AstNodeVisitor &p_visitor) override;
    
    const char* getVariableName();

  private:
    // TODO: variable name, expressions
    std::string m_variable_name;
    std::vector<ExpressionNode*> *m_expressions;
};

#endif
