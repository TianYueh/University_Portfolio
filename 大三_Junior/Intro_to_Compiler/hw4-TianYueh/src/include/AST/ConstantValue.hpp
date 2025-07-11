#ifndef AST_CONSTANT_VALUE_NODE_H
#define AST_CONSTANT_VALUE_NODE_H

#include "AST/PType.hpp"
#include "AST/constant.hpp"
#include "AST/expression.hpp"
#include "visitor/AstNodeVisitor.hpp"

#include <memory>

class ConstantValueNode final : public ExpressionNode {
  private:
    std::unique_ptr<Constant> m_constant_ptr;

  public:
    ~ConstantValueNode() = default;
    ConstantValueNode(const uint32_t line, const uint32_t col,
                      Constant *const p_constant)
        : ExpressionNode{line, col}, m_constant_ptr(p_constant) {
          type = p_constant->getTypeSharedPtr();
        }

    const PTypeSharedPtr &getTypeSharedPtr() const;
    

    const char *getConstantValueCString() const;

    void accept(AstNodeVisitor &p_visitor) override { p_visitor.visit(*this); }
};

#endif
