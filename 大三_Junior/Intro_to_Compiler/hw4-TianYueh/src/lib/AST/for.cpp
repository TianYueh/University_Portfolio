#include "AST/for.hpp"

void ForNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    m_loop_var_decl->accept(p_visitor);
    m_init_stmt->accept(p_visitor);
    m_end_condition->accept(p_visitor);
    m_body->accept(p_visitor);
}

const char* ForNode::getInitValCString() {
    return m_init_stmt->getConstantValueCString();
}

const char* ForNode::getConditionValCString() {
    std::unique_ptr<ConstantValueNode> const_val(dynamic_cast<ConstantValueNode*>(m_end_condition.get()));
    if(const_val){
        m_end_condition.release();
    }
    
    return const_val->getConstantValueCString();
}
