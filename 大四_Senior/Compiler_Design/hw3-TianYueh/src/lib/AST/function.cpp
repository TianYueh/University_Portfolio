#include "AST/function.hpp"

// TODO
FunctionNode::FunctionNode(const uint32_t line, const uint32_t col,
                           std::string function_name,
                           std::vector<DeclNode*> *p_declarations,
                           std::string return_type,
                           CompoundStatementNode *p_compound_statement)
    : AstNode{line, col},
      m_function_name(function_name),
      m_declarations(p_declarations),
      m_return_type(return_type),
      m_compound_statement(p_compound_statement) {
        std::string tmp_var_type;
        if (m_declarations != nullptr){
            for(int i = 0; i < m_declarations->size() - 1; i++){
                std::vector<const char*> decltypes = m_declarations->at(i)->getDeclType();
                for (int j = 0; j < decltypes.size(); j++){
                    tmp_var_type += decltypes[j];
                    tmp_var_type += ", ";
                }
            }
            std::vector<const char*> decltypes = m_declarations->at(m_declarations->size() - 1)->getDeclType();

            for (int j = 0; j < decltypes.size() - 1; j++){
                tmp_var_type += decltypes[j];
                tmp_var_type += ", ";
            }
            tmp_var_type += decltypes[decltypes.size() - 1];
        }
        m_parameter_type = return_type + " (" + tmp_var_type + ")";
      }

// TODO: You may use code snippets in AstDumper.cpp
void FunctionNode::print() {}

void FunctionNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    // TODO
    if (m_declarations != nullptr) {
        for (auto decl : *m_declarations) {
            decl->accept(p_visitor);
        }
    }
    if (m_compound_statement != nullptr) {
        m_compound_statement->accept(p_visitor);
    }
    
}
