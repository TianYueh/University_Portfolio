#ifndef AST_PROGRAM_NODE_H
#define AST_PROGRAM_NODE_H

#include "AST/ast.hpp"

#include "AST/CompoundStatement.hpp"
#include "visitor/AstNodeVisitor.hpp"
#include "AST/decl.hpp"
#include "AST/function.hpp"
#include <memory>
#include <vector>
#include <string>

class ProgramNode final : public AstNode {
  private:
    std::string name;
    // TODO: return type, declarations, functions, compound statement
    //const char *const m_return_type;
    std::vector<DeclNode*>* m_var_decls;
    std::vector<FunctionNode*>* m_func_decls;

  
    CompoundStatementNode* m_body;

  public:
    ~ProgramNode() = default;
    ProgramNode(const uint32_t line, const uint32_t col,
                const char *const p_name,
                //const char *const p_return_type,
                std::vector<DeclNode*> * const p_var_decls,
                std::vector<FunctionNode*> * const p_func_decls,

                CompoundStatementNode * const p_body
                /* TODO: return type, declarations, functions,
                 *       compound statement */
                
                );

    // visitor pattern version: const char *getNameCString() const;
    void print() override;
    const char *getNameCString() const { return name.c_str(); }
    void accept(AstNodeVisitor &p_visitor) override { p_visitor.visit(*this); }
    void visitChildNodes(AstNodeVisitor &p_visitor) override;
};

#endif
