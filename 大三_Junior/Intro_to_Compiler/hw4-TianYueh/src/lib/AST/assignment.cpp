#include "AST/assignment.hpp"
#include "AST/ConstantValue.hpp"

void AssignmentNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    m_lvalue->accept(p_visitor);
    m_expr->accept(p_visitor);
}

const char* AssignmentNode::getConstantValueCString() {
    std::unique_ptr<ConstantValueNode> const_val(dynamic_cast<ConstantValueNode *>(m_expr.get()));
    if(const_val){
        m_expr.release();
    }
    return const_val->getConstantValueCString();
}

const char* AssignmentNode::getLvalueNameCString() {
    return m_lvalue->getNameCString();
}

const char* AssignmentNode::getLvalueTypeCString() {
    return m_lvalue->getPTypeCString();
}

const char* AssignmentNode::getRvalueTypeCString() {
    return m_expr->getPTypeCString();
}

const uint32_t AssignmentNode::getLvalueLocationCol() const {
    return m_lvalue->getLocation().col;
}

const uint32_t AssignmentNode::getRvalueLocationCol() const {
    return m_expr->getLocation().col;
}

const int AssignmentNode::checkInvalidLvalue() const {
    std::string lvalue = m_lvalue->getPTypeCString();
    if(lvalue == "null"){
        return 1;
    }
    else{
        return 0;
    }
}

const int AssignmentNode::checkInvalidRvalue() const {
    std::string rvalue = m_expr->getPTypeCString();
    if(rvalue == "null"){
        return 1;
    }
    else{
        return 0;
    }
}

const int AssignmentNode::checkLvalueScalarType() const {
    std::string lvalue = m_lvalue->getPTypeCString();
    if(lvalue == "integer" || lvalue == "real" || lvalue == "boolean" || lvalue == "string"){
        return 1;
    }
    else{
        return 0;
    }
}

const int AssignmentNode::checkRvalueScalarType() const {
    std::string rvalue = m_expr->getPTypeCString();
    if(rvalue == "integer" || rvalue == "real" || rvalue == "boolean" || rvalue == "string"){
        return 1;
    }
    else{
        return 0;
    }
}

const int AssignmentNode::checkCompatibleLRvalueType() const {
    std::string lval = m_lvalue->getPTypeCString();
    std::string rval = m_expr->getPTypeCString();
    if(lval == rval){
        return 1;
    }
    else if(lval == "real" && rval == "integer"){
        return 1;
    }
    else{
        return 0;
    }
}


