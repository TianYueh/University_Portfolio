#include "AST/FunctionInvocation.hpp"

#include <algorithm>

void FunctionInvocationNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    auto visit_ast_node = [&](auto &ast_node) { ast_node->accept(p_visitor); };

    for_each(m_args.begin(), m_args.end(), visit_ast_node);
}

const int FunctionInvocationNode::getArgNum() const {
    return m_args.size();
}

const uint32_t FunctionInvocationNode::getArgLocationCol(const int id) const {
    return m_args[id]->getLocation().col;
}

checkArgRet* FunctionInvocationNode::checkArgType(const char* attr) const {
    checkArgRet* ret = new checkArgRet;
    std::string attr_str(attr);
    bool isWrong = false;
    size_t l = 0;
    size_t r = attr_str.find(',');
    for(int i = 0; i<m_args.size(); i++){
        if(r == std::string::npos){
            r = attr_str.size();
        }
        std::string arg = m_args[i]->getPTypeCString();
        std::string p = attr_str.substr(l, r-l);

        if(p == "real"){
            if(arg != "integer" && arg != "real"){
                ret->wrongid = i;
                ret->p = p;
                ret->arg = arg;
                isWrong = true;
                break;
            }
        }
        else if(p!=arg){
            ret->wrongid = i;
            ret->p = p;
            ret->arg = arg;
            isWrong = true;
            break;
        }

        if(i != m_args.size()-1){
            attr_str = attr_str.substr(r+2);
            r = attr_str.find(',', l);
        }
    }

    if(!isWrong){
        ret->wrongid = -1;
    }
    return ret;
}

const int FunctionInvocationNode::CheckInvalidChildren() const {
    for (auto &arg : m_args) {
        if (arg->getPTypeCString() == "null") {
            return 1;
        }
    }
    return 0;
}
