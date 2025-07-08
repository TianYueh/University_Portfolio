#ifndef AST_EXPRESSION_NODE_H
#define AST_EXPRESSION_NODE_H

#include "AST/ast.hpp"
#include "AST/PType.hpp"

class ExpressionNode : public AstNode {
  public:
    ~ExpressionNode() = default;
    ExpressionNode(const uint32_t line, const uint32_t col)
        : AstNode{line, col} {}

    PType *type_ptr = new PType(PType::PrimitiveTypeEnum::kErrorType);
    virtual int getSizeOfDimension() {
        return 0;
    } 

  //protected:
    // for carrying type of result of an expression
    // TODO: for next assignment


};

#endif
