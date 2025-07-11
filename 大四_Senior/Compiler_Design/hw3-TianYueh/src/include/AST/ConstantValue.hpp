#ifndef __AST_CONSTANT_VALUE_NODE_H
#define __AST_CONSTANT_VALUE_NODE_H

#include "AST/expression.hpp"
#include "visitor/AstNodeVisitor.hpp"
#include <string>

class ConstantValueNode : public ExpressionNode {
  public:
    ConstantValueNode(const uint32_t line, const uint32_t col,
                      const char *p_constant_value
                      /* TODO: constant value */);
    ~ConstantValueNode() = default;

    void print() override;
    void accept(AstNodeVisitor &p_visitor) override {
        p_visitor.visit(*this);
    }
    const char* getConstantValue();

  private:
    // TODO: constant value
    std::string m_constant_value;
};

#endif
