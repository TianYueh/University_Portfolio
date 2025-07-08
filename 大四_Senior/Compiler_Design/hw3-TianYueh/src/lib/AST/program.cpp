#include "AST/program.hpp"


// TODO
ProgramNode::ProgramNode(const uint32_t line, const uint32_t col,
                         const char *const p_name, 
                         //const char *const p_return_type,
                         std::vector<DeclNode*> * const p_var_decls,
                         std::vector<FunctionNode*> * const p_func_decls,
                         CompoundStatementNode *const p_body)
    : AstNode{line, col}, name(p_name), 
      //m_return_type(p_return_type),
      m_var_decls(p_var_decls),
      m_func_decls(p_func_decls),
      m_body(p_body) {}

// visitor pattern version: const char *ProgramNode::getNameCString() const { return name.c_str(); }

void ProgramNode::print() {
    // TODO
    // outputIndentationSpace();

    std::printf("program <line: %u, col: %u> %s %s\n",
                location.line, location.col,
                name.c_str(), "void");

    // TODO
    // incrementIndentation();
    // visitChildNodes();
    // decrementIndentation();
}


void ProgramNode::visitChildNodes(AstNodeVisitor &p_visitor) { // visitor pattern version
     /* TODO
      *
      * for (auto &decl : var_decls) {
      *     decl->accept(p_visitor);
      * }
      *
      * // functions
      *
      * body->accept(p_visitor);
      */
 
    if (m_var_decls != nullptr) {
        for (auto &decl : *m_var_decls) {
            decl->accept(p_visitor);
        }
    }
    if (m_func_decls != nullptr) {
        for (auto &func : *m_func_decls) {
            func->accept(p_visitor);
        }
    }
    
    m_body->accept(p_visitor);
}
