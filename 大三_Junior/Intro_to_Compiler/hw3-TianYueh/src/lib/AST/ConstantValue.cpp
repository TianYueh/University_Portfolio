#include "AST/ConstantValue.hpp"
#include <string.h>

char rtnstr[100];
// TODO
ConstantValueNode::ConstantValueNode(const uint32_t line, const uint32_t col,
                                    Constant_Value p_const_val)
    : ExpressionNode{line, col}, const_val(p_const_val) {}

// TODO: You may use code snippets in AstDumper.cpp
void ConstantValueNode::print() {}

void ConstantValueNode::accept(AstNodeVisitor &p_visitor){
    p_visitor.visit(*this);
}

const char* ConstantValueNode::getCStringValue() const{
    //memset(rtnstr, 0 , 100);
    if(const_val.bool_type){
        return const_val.str_value;
    }
    else if(const_val.real_type){
        sprintf(rtnstr, "%f", const_val.real_value);
    }
    else if(const_val.int_type){
        sprintf(rtnstr, "%d", const_val.int_value);
    }
    else if(const_val.str_type){
        return const_val.str_value;
    }

    return rtnstr;
}
