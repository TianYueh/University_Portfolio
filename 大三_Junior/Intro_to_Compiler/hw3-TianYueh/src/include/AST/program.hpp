#ifndef AST_PROGRAM_NODE_H
#define AST_PROGRAM_NODE_H

#include "AST/ast.hpp"
#include "AST/CompoundStatement.hpp"
#include "visitor/AstNodeVisitor.hpp"

#include <memory>
#include <string>

class ProgramNode final : public AstNode {
  private:
    std::string name;
    const std::string rtn_type;
    std::vector<AstNode*> *decl_list;
    std::vector<AstNode*> *func_list;
    AstNode *m_body; 
    
    // TODO: return type, declarations, functions, compound statement

  public:
    ~ProgramNode() = default;
    ProgramNode(const uint32_t line, const uint32_t col,
                const char* p_name, 
                const char* p_rtn_type,
                std::vector<AstNode*> *p_decl_list,
                std::vector<AstNode*> *p_func_list,
                AstNode *const p_body
                /* TODO: return type, declarations, functions,
                 *       compound statement */);

    // visitor pattern version: const char *getNameCString() const;
    void print() override;
    const char *getNameCString() const;
    void accept(AstNodeVisitor &p_visitor) override ;
    void visitChildNodes(AstNodeVisitor &p_visitor) override;
};

#endif
