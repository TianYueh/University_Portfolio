#ifndef SEMA_SEMANTIC_ANALYZER_H
#define SEMA_SEMANTIC_ANALYZER_H

#include "visitor/AstNodeVisitor.hpp"
#include <vector>
#include <string>

#include <stack>
//Symbol table also implemented here
class SymbolEntry{
  public:
    SymbolEntry(const char* p_varName, const char* p_kind, const int p_level, const char* p_type, const char* p_attr):
      varName(p_varName), kind(p_kind), level(p_level), type(p_type), attr(p_attr){}

    void dumpEntry(void);
    const char* getNameCString() const;
    const char* getKindCString() const;
    const char* getTypeCString() const;
    const char* getAttrCString() const;
    int getTypeDim() const;
    void setKindString(const char* p_kind);
    void setAttrString(const char* p_attr);
    void getNewTypeDims(std::vector<uint64_t> &dims, int ignored) const;
    int getFunctionParamNum() const;
  private:
    std::string varName;
    std::string kind;
    int level;
    std::string type;
    std::string attr;

};

class SymbolTable{
  public:
    SymbolTable(){}
    void addSymbol(SymbolEntry* p_entry);
    void dumpSymbolTable(void);
    int checkRedecl(const char* p_varName) const;
    void addError(const char* p_name);
    int checkErrDecl(const char* p_name) const;
    SymbolEntry* getSymbolEntry(const char* p_varName);

  private:
    std::vector<SymbolEntry*> symbol_table;
    std::vector<const char*> errDecl;
};

class SymbolManager{
  public:
    SymbolManager(){}
    SymbolTable* getTopSymbolTable();
    void pushScope(SymbolTable* newScope);
    void popScope(void);
    int checkLoopVarRedecl(const char*);
    int checkConst(const char*);
    void push_loop_var(const char* p_varName);
    void pop_loop_var(void);
    void push_const(const char* p_varName);
    void pop_const(void);

    int getScopeSize(void);


  private:
    std::stack<SymbolTable*> symbol_table_stack;
    std::vector<const char*> loop_var;
    std::vector<const char*> consts;
};

class SemanticAnalyzer final : public AstNodeVisitor {
  private:
    // TODO: something like symbol manager (manage symbol tables)
    //       context manager, return type manager
    bool isfor = false;
    bool isloop_var = false;
    bool isFunc = false;
    bool isInFunc = false;
    bool isVar = false;
    int m_level = 0;

    std::string cur_func_name;
    SymbolManager* symbol_manager;
    SymbolManager* tmp_manager;
    SymbolTable* cur_symbol_table;
    SymbolEntry* cur_symbol_entry;
    
  public:
    ~SemanticAnalyzer() = default;
    SemanticAnalyzer() = default;

    void visit(ProgramNode &p_program) override;
    void visit(DeclNode &p_decl) override;
    void visit(VariableNode &p_variable) override;
    void visit(ConstantValueNode &p_constant_value) override;
    void visit(FunctionNode &p_function) override;
    void visit(CompoundStatementNode &p_compound_statement) override;
    void visit(PrintNode &p_print) override;
    void visit(BinaryOperatorNode &p_bin_op) override;
    void visit(UnaryOperatorNode &p_un_op) override;
    void visit(FunctionInvocationNode &p_func_invocation) override;
    void visit(VariableReferenceNode &p_variable_ref) override;
    void visit(AssignmentNode &p_assignment) override;
    void visit(ReadNode &p_read) override;
    void visit(IfNode &p_if) override;
    void visit(WhileNode &p_while) override;
    void visit(ForNode &p_for) override;
    void visit(ReturnNode &p_return) override;
};

#endif
