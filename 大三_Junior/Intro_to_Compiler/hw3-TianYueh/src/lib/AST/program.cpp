#include "AST/program.hpp"

// TODO
ProgramNode::ProgramNode(const uint32_t line, const uint32_t col,
                        const char *const p_name,
                        const char* p_rtn_type,
                        std::vector<AstNode*> *p_decl_list,
                        std::vector<AstNode*> *p_func_list,
                        AstNode* p_body)
    : AstNode{line, col}, name(p_name), rtn_type(p_rtn_type), decl_list(p_decl_list), func_list(p_func_list), m_body(p_body) {}

const char *ProgramNode::getNameCString() const { return name.c_str(); }

void ProgramNode::print() {
    // TODO
    // outputIndentationSpace();
    /*


    std::printf("program <line: %u, col: %u> %s %s\n",
                location.line, location.col,
                name.c_str(), "void");

    */
    // TODO
    // incrementIndentation();
    // visitChildNodes();
    // decrementIndentation();
}

void ProgramNode::accept(AstNodeVisitor &p_visitor){
    p_visitor.visit(*this);
}


void ProgramNode::visitChildNodes(AstNodeVisitor &p_visitor) { // visitor pattern version
//     /* TODO
//      *
//      * for (auto &decl : var_decls) {
//      *     decl->accept(p_visitor);
//      * }
//      *
//      * // functions
//      *
        if(decl_list != NULL){
            for(auto &decl : *decl_list){
                decl->accept(p_visitor);
            }
        }
        if(func_list != NULL){
            for(auto &func : *func_list){
                func->accept(p_visitor);
            }
        }
        if(m_body != NULL){
            m_body->accept(p_visitor);
        }
        
}
