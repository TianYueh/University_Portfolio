#ifndef __AST_VARIABLE_NODE_H
#define __AST_VARIABLE_NODE_H

#include "AST/ast.hpp"
#include "AST/ConstantValue.hpp"
#include "visitor/AstNodeVisitor.hpp"
#include <string>
#include <vector>

class VariableNode : public AstNode {
  public:
    VariableNode(const uint32_t line, const uint32_t col,
                 /* TODO: variable name, type, constant value */
                const char* p_variable_name,
                const char* p_variable_type,
                ConstantValueNode *p_constant_value);
    ~VariableNode() = default;

    void print() override;
    void accept(AstNodeVisitor &p_visitor) override {
        p_visitor.visit(*this);
    }
    void visitChildNodes(AstNodeVisitor &p_visitor) override;
    const char* getVariableName() {
        return m_variable_name.c_str();
    }
    const char* getVariableType() {
        return m_variable_type.c_str();
    }
    

  private:
    // TODO: variable name, type, constant value
    std::string m_variable_name;
    std::string m_variable_type;
    ConstantValueNode *m_constant_value;
};

#endif
