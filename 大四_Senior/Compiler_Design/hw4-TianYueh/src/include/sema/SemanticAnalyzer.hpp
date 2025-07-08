#ifndef SEMA_SEMANTIC_ANALYZER_H
#define SEMA_SEMANTIC_ANALYZER_H

#include "sema/ErrorPrinter.hpp"
#include "visitor/AstNodeVisitor.hpp"

#include "AST/PType.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <stack>
#include <iomanip>
#include <fstream>
#include <sstream>
class SymbolEntry{
  private:
      std::string name;
      std::string kind;
      int level;
      std::string type;
      PType *type_ptr;
      std::string attribute;

      std::vector<PType*> params_type;

      int integer_value;
      float real_value;
      bool bool_value;
      std::string string_value;

      bool isDeclError = false;
      int dims;
      
  public:
    SymbolEntry(std::string p_name, std::string p_kind, int p_level, std::string p_type, PType *p_type_ptr)
      : name(p_name), kind(p_kind), level(p_level), type(p_type), type_ptr(p_type_ptr){
      dims = 0;
      for (auto &c : type){
        if(c == '['){
          dims++;
        }
      }
    }

    SymbolEntry(std::string p_name, std::string p_kind, int p_level, std::string p_type, std::string p_attribute, PType *p_type_ptr, std::vector<PType*> p_parameters_type)
      : name(p_name), kind(p_kind), level(p_level), type(p_type), attribute(p_attribute), type_ptr(p_type_ptr), params_type(p_parameters_type){ 
      dims = 0;
        for (auto &c : type){
          if(c == '['){
            dims++;
          }
        }
      }
    
    SymbolEntry(std::string p_name, std::string p_kind, int p_level, std::string p_type, bool p_isDeclError, PType *p_type_ptr)
      : name(p_name), kind(p_kind), level(p_level), type(p_type), isDeclError(p_isDeclError), type_ptr(p_type_ptr){ 
        dims = 0;
        for (auto &c : type){
          if(c == '['){
            dims++;
          }
        }
    }
              
    // Print the entry in the formatted way
    void PrintEntry(){
      std::string string_level;
      if(level == 0){
        string_level = std::to_string(level) + "(global)";
      }
      else{
        string_level = std::to_string(level) + "(local)";
      }
      printf("%-33s%-11s%-11s%-17s%-11s\n", name.c_str(), kind.c_str(), string_level.c_str(), type.c_str(), attribute.c_str());
    }

    void addConstant(std::string val){
      kind = "constant";
      attribute = val;
      if(type == "integer"){
        integer_value = stoi(attribute, NULL, 10);
      }
      else if(type == "real"){
        real_value = stof(attribute);
      }
      else if(type == "string"){
        string_value = attribute;
      }
      else if(type == "boolean"){
        if (attribute == "true"){
          bool_value = true;
        }
        else{
          bool_value = false;
        }
      }
    }

    PType *getType(std::string cur){
      if (cur == name){
        return type_ptr;
      }
      return nullptr;
    }

    std::pair<std::vector<PType*>, bool> getParameterType(std::string cur){
      std::vector<PType*> para_type;
      bool isFunction = false;
      if (cur == name && kind == "function"){
        isFunction = true;
        std::pair<std::vector<PType*>, bool> rtn_type(params_type, isFunction);
        return rtn_type;
      }
      else{
        std::pair<std::vector<PType*>, bool> rtn_type(para_type, isFunction);
        return rtn_type;
      }
    }

    std::string getKind(std::string cur){ 
      if (cur == name){
          return kind;
        }
      return "";
    }

    bool VarRedecl(std::string cur){ 
      if(cur == name){
          return true;
        }
      return false;
    }
    bool LoopVarRedecl(std::string cur){ 
      if(cur == name && kind == "loop_var"){
          return true;
        }
      return false;
    }
    bool VarKind(std::string cur){ 
      if(cur == name && ( kind == "parameter" || kind == "variable" || kind == "loop_var" || kind == "constant" )){
          return true;
        }
      return false;
    }
    bool NonFunctionSymbol(std::string cur){ 
      if(cur == name && kind == "function"){
          return true;
        }
      return false;
    }
    bool OverArraySubscript(std::string cur_name, int cur_sz){ 
      if(cur_name == name && cur_sz > dims){
          return true;
        }
      return false;
    }
    bool WrongDeclOrNot(std::string cur){ 
      if(cur == name){
          return isDeclError;
        }
      return false;
    }
    bool ParameterOver(std::string cur, int cur_sz){ 
      if (cur == name && cur_sz != params_type.size()){
          return true;
        }
      return false;
    }
};

class SymbolTable{
  public:
    SymbolTable(int p_level)
              : level(p_level){}
    SymbolTable(int p_level, std::string p_func_name)
              : level(p_level), func_name(p_func_name){}

    void addSymbol(std::string p_name, std::string p_kind, int p_level, std::string p_type, PType *p_type_ptr){
      SymbolEntry symbol_entry(p_name, p_kind, p_level, p_type, p_type_ptr);
      entries.push_back(symbol_entry);
    }
    void addSymbol(std::string p_name, std::string p_kind, int p_level, std::string p_type, std::string p_attribute, PType *p_type_ptr, std::vector<PType*> p_parameters_type){
      SymbolEntry symbol_entry(p_name, p_kind, p_level, p_type, p_attribute, p_type_ptr, p_parameters_type);
      entries.push_back(symbol_entry);
    }
    void addSymbol(std::string p_name, std::string p_kind, int p_level, std::string p_type, bool p_declaration_error, PType *p_type_ptr){
      SymbolEntry symbol_entry(p_name, p_kind, p_level, p_type, p_declaration_error, p_type_ptr);
      entries.push_back(symbol_entry);
    }

    void PrintTable(){
      for(auto &e : entries){
        e.PrintEntry();
      }
    }

    void addConstantEntry(std::string c){
      entries.back().addConstant(c);
    }

    int getLevel(){ 
      return level; 
    }

    PType *getType(std::string cur){
      for(auto &e : entries){
        if(e.getType(cur) != nullptr){
          return e.getType(cur);
        }
      }
      return nullptr;
    }

    std::pair<std::vector<PType*>, bool> getParameterType(std::string cur){
      std::vector<PType*> new_types;
      for(auto &e : entries){
        if(e.getParameterType(cur).second){
          return e.getParameterType(cur);
        }
      }
      std::pair<std::vector<PType*>, bool> new_types_pair(new_types, false);
      return new_types_pair;
    }

    std::string getKind(std::string cur){
      for(auto &e : entries){
        if(e.getKind(cur) != ""){
          return e.getKind(cur);
        }
      }
      return "";
    }

    bool VarRedecl(std::string cur){
      for(auto &e : entries){
        if(e.VarRedecl(cur)){
          return true;
        }
      }
      return false;
    }

    bool LoopVarRedecl(std::string cur){
      for(auto &e : entries){
        if(e.LoopVarRedecl(cur)){
          return true;
        }
      }
      return false;
    }

    bool VarKind(std::string cur){
      for(auto &e : entries){
        if(e.VarKind(cur)){
          return true;
        }
      }
      return false;
    }
    
    bool NonFunctionSymbol(std::string cur){
      for(auto &e : entries){
        if(e.NonFunctionSymbol(cur)){
          return true;
        }
      }
      return false;
    }
    
    bool OverArraySubscript(std::string cur_name, int cur_dim){
      for(auto &e : entries){
        if(e.OverArraySubscript(cur_name, cur_dim)){
          return true;
        }
      }
      return false;
    }
    
    bool WrongDeclOrNot(std::string cur){
      for(auto &e : entries){
        if(e.WrongDeclOrNot(cur)){
          return true;
        }
      }
      return false;
    }

    bool ParameterOver(std::string cur_name, int cur_sz){
      for(auto &e : entries){
        if(e.ParameterOver(cur_name, cur_sz)){
          return true;
        }
      }
      return false;
    }
      
  private:
    std::vector<SymbolEntry> entries;
    int level;
    std::string func_name;
};

class SymbolManager{

  public:
    std::vector<std::string> cur_root;
    void pushScope(SymbolTable *new_scope){
      tables.push_back(new_scope); 
      level++;
    }
    void popScope(){ 
      tables.pop_back();
      level--; 
    }

    int getLevel(){ 
      return level; 
    }
    SymbolTable* getTableTop(){ 
      return tables.back(); 
    }

    PType *getType(std::string cur){
      for(int i = tables.size() - 1; i >= 0; i--){
        if(tables[i]->getType(cur) != nullptr){
          return tables[i]->getType(cur);
        }
      }
      return nullptr;
    }

    std::string getKind(std::string cur){
      for(int i = tables.size() - 1; i >= 0; i--){
        if(tables[i]->getKind(cur) != ""){
          return tables[i]->getKind(cur);
        }
      }
      return "";
    }

    PType *getFunctionType(std::string cur){
      for(int i = tables.size() - 1; i >= 0; i--){
        if(tables[i]->getType(cur) != nullptr && NonFunctionSymbolError(cur)){
          return tables[i]->getType(cur);
        }
      }
      return nullptr;
    }
    
    bool RedeclarationError(std::string var){
      for(auto &t : tables){
        if(t->LoopVarRedecl(var)){
            return true;
        }
      }
      return tables.back()->VarRedecl(var);
    }

    bool UndeclaredError(std::string cur){
      for(auto &t : tables){
        if(t->VarRedecl(cur)){
          return true;
        }
      }
      return false;
    }

    bool NonVariableError(std::string cur){
      for(auto &t : tables){
        if(t->VarKind(cur)){
          return true;
        }
      }
      return false;
    }

    bool NonFunctionSymbolError(std::string cur){
      for(auto &t : tables){
        if(t->NonFunctionSymbol(cur)){
          return true;
        }
      }
      return false;
    }

    bool OverArraySubscriptError(std::string cur_name, int cur_dim){
      for(auto &t : tables){
        if(t->OverArraySubscript(cur_name, cur_dim)){
          return true;
        }
      }
      return false;
    }

    bool WrongDeclOrNot(std::string cur){
      for(auto &t : tables){
        if(t->WrongDeclOrNot(cur)){
          return true;
        }
      }
      return false;
    }

    bool ArgumentNumberMismatchError(std::string cur_name, int cur_sz){
      for(auto &t:tables){
        if(t->ParameterOver(cur_name, cur_sz)){
          return true;
        }
      }
      return false;
    }

    PType *getParameterType(std::string func_name, int param_i){
      for(auto &t:tables){
        if(t->getParameterType(func_name).second){
          return t->getParameterType(func_name).first[param_i];
        }
      }
    }

    bool LoopVarOrNot(std::string cur){
      for(auto &t:tables){
        if(t->LoopVarRedecl(cur)){
          return true;
        }
      }
      return false;
    }
  private:
    std::vector<SymbolTable*> tables;
    int level = 0;
};

class ContextManager{
  public:
    int param_num = 0;
	  bool isParamError = false;
	  std::vector<PType*> expr_type;
	  std::vector<std::string> func_name;
};

class ReturnTypeManager{
  public:
	  std::vector<std::string> func_in;
	  std::string getReturnTypeString(int dim_num, PType *type){
	  	std::string rtn_type = type->getType();
	  	if(dim_num < type->getDimensionsSize()){
	  		rtn_type = rtn_type + " ";
	  	}
	  	for(int i = dim_num; i < type->getDimensionsSize(); i++){
	  		rtn_type = rtn_type + "[" + std::to_string(type->getDimensions()[i]) + "]";
	  	}
	  	return rtn_type;
	  }
};

class SemanticAnalyzer final : public AstNodeVisitor{
  private:
    ErrorPrinter m_error_printer{stderr};
    // TODO: something like symbol manager (manage symbol tables)
    //       context manager, return type manager
    SymbolManager symbolManager;
    ContextManager contextManager;
    ReturnTypeManager returnTypeManager;
    
    bool isPrintError = false;
    bool isReadError = false;
    bool isIfError = false;
    bool isWhileError = false;
	  bool isConstantError = false;

  public:
    int loopStart = 0;
    int loopEnd = 0;
    bool isForError(){
      if (loopEnd < loopStart){
        return true;
      }
      return false;
    }

    bool isErrorBFlag = false;
    std::vector<PType*> assignmentTypeVector;


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
