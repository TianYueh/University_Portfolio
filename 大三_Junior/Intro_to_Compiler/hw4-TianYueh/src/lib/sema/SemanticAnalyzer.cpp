#include "sema/SemanticAnalyzer.hpp"
#include "visitor/AstNodeInclude.hpp"
#include "AST/PType.hpp"
#include <iostream>

#include <string.h>

extern uint32_t opt_d;
extern bool isError;
extern char* source_code[512];

/* Demarcation */

static void dumpDemarcation(const char chr){
    for(int i = 0;i < 110 ;i++){
        printf("%c", chr);
    }
    puts("");
}

/* Error Message */

static void showErrorMessage(uint32_t line, uint32_t col, std::string msg){
    std::cerr << "<Error> Found in line " << line << ", column " << col << ": " << msg << std::endl;
    std::cerr << "    " <<source_code[line]<< std::endl;
    for (int i = 0; i < col+3; i++){
        std::cerr << " ";
    }
    std::cerr<< "^" << std::endl;
}


/* Implement the Symbol Entry */

void SymbolEntry::dumpEntry(void){
  std::string what_dump;
  //printf("%d", level);
  what_dump = (level == 0) ? "(global)" : "(local)";
  printf("%-33s%-11s%d%-10s%-17s%-11s\n", varName.c_str(), kind.c_str(), level, what_dump.c_str(), type.c_str(), attr.c_str());

}

const char* SymbolEntry::getNameCString() const{
  return varName.c_str();
}

const char* SymbolEntry::getKindCString() const{
  return kind.c_str();
}

const char* SymbolEntry::getTypeCString() const{
  return type.c_str();
}

const char* SymbolEntry::getAttrCString() const{
  return attr.c_str();
}


int SymbolEntry::getTypeDim() const{
  int dim = 0;
  for(int i = 0; i < type.length(); i++){
    if(type[i] == '['){
        dim++;
    }
  }
  return dim;
}

void SymbolEntry::setKindString(const char* p_kind){
  kind = p_kind;
}

void SymbolEntry::setAttrString(const char* p_attr){
  attr = p_attr;
}

void SymbolEntry::getNewTypeDims(std::vector<uint64_t> &dims, int ignored) const{
    int cnt = 0;

    for(int i = 0;i < type.size(); i++){
        if(type[i] != '['){
            continue;
        }
        if(cnt != ignored){
            cnt++;
            continue;
        }
        uint64_t dim = 0;
        i++;
        while(type[i] != ']'){
            dim = dim * 10 + (type[i] - '0');
            i++;
        }
        dims.push_back(dim);
    }
}

int SymbolEntry::getFunctionParamNum() const{
    int cnt = 0;
    if(attr.size() == 0){
        return 0;
    }
    else{
        cnt = 1;
    }
    for(int i = 0; i < attr.size(); i++){
        if(attr[i] == ','){
            cnt++;
        }
    }

    return cnt;
}

/* Implement the Symbol Table */

void SymbolTable::addSymbol(SymbolEntry* p_entry){
  symbol_table.push_back(p_entry);
}

void SymbolTable::dumpSymbolTable(void){
  dumpDemarcation('=');  
  printf("%-33s%-11s%-11s%-17s%-11s\n", "Name", "Kind", "Level", "Type", "Attribute");
  dumpDemarcation('-');
  
  for(int i = 0; i < symbol_table.size(); i++){
    symbol_table[i]->dumpEntry();
  }
  dumpDemarcation('-');
}


int SymbolTable::checkRedecl(const char* p_varName) const{
  for(int i = 0; i < symbol_table.size(); i++){
    if((std::string)symbol_table[i]->getNameCString() == p_varName){
      return 1;
    }
  }
  return 0;
}





void SymbolTable::addError(const char* p_name){
  errDecl.push_back(p_name);
}

int SymbolTable::checkErrDecl(const char* p_name) const{
  for(int i = 0; i < errDecl.size(); i++){
    if((std::string)errDecl[i] == p_name){
      return 1;
    }
  }
  return 0;
}

SymbolEntry* SymbolTable::getSymbolEntry(const char* p_varName){
  for(int i = 0; i < symbol_table.size(); i++){
    if((std::string)symbol_table[i]->getNameCString() == p_varName){
      return symbol_table[i];
    }
  }
  return nullptr;
}

/* Implement the Symbol Manager */

SymbolTable* SymbolManager::getTopSymbolTable(void){
  return symbol_table_stack.top();
}

void SymbolManager::pushScope(SymbolTable* newScope){
  symbol_table_stack.push(newScope);
}

void SymbolManager::popScope(void){
  symbol_table_stack.pop();
}

int SymbolManager::checkLoopVarRedecl(const char* p_varName){
  for(int i = 0; i < loop_var.size(); i++){
    if((std::string)loop_var[i] == p_varName){
      return 1;
    }
  }
  return 0;
}

int SymbolManager::checkConst(const char* p_name){
  for(int i = 0; i < consts.size(); i++){
    if((std::string)consts[i] == p_name){
      return 1;
    }
  }
  return 0;
}

void SymbolManager::push_loop_var(const char* p_varName){
  loop_var.push_back(p_varName);
}

void SymbolManager::pop_loop_var(void){
  loop_var.pop_back();
}

void SymbolManager::push_const(const char* p_const){
  consts.push_back(p_const);
}

void SymbolManager::pop_const(void){
  consts.pop_back();
}

int SymbolManager::getScopeSize(void){
  return symbol_table_stack.size();
}


/* Implement the Semantic Analyzer */

void SemanticAnalyzer::visit(ProgramNode &p_program) {
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Travere child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */

    symbol_manager = new SymbolManager();
    tmp_manager = new SymbolManager();
    cur_symbol_table = new SymbolTable();
    cur_symbol_entry = new SymbolEntry(p_program.getNameCString(), "program", m_level, "void", "");
    
    cur_symbol_table->addSymbol(cur_symbol_entry);
    symbol_manager->pushScope(cur_symbol_table);

    p_program.visitChildNodes(*this);
    if(opt_d){
        cur_symbol_table = symbol_manager->getTopSymbolTable();
        cur_symbol_table->dumpSymbolTable();
    }
    symbol_manager->popScope();

}

//Checkpoint

void SemanticAnalyzer::visit(DeclNode &p_decl) {
    p_decl.visitChildNodes(*this);
}

void SemanticAnalyzer::visit(VariableNode &p_variable) {
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Travere child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */

    cur_symbol_table = symbol_manager->getTopSymbolTable();
    bool arufa = cur_symbol_table->checkRedecl(p_variable.getNameCString());
    bool beta = symbol_manager->checkLoopVarRedecl(p_variable.getNameCString());

    if(arufa || beta){
        isError = true;
        std::string msg = "symbol \'" + std::string(p_variable.getNameCString()) + "\' is redeclared";
        showErrorMessage(p_variable.getLocation().line, p_variable.getLocation().col, msg);
    }
    else{
        std::string type;
        if(isFunc){
            type = "parameter";
        }
        else if(isfor){
            type = "loop_var";
            isfor = false;
            symbol_manager->push_loop_var(p_variable.getNameCString());
        }
        else{
            type = "variable";
        }

        cur_symbol_entry = new SymbolEntry(p_variable.getNameCString(), type.c_str(), m_level, p_variable.getTypeCString(), "");
        cur_symbol_table->addSymbol(cur_symbol_entry);

        isVar = true;
        p_variable.visitChildNodes(*this);
        if((std::string)cur_symbol_entry->getKindCString() == "constant"){
            symbol_manager->push_const(p_variable.getNameCString());
        }

        isVar = false;

    }
    if(p_variable.checkInvalidDim()){
        isError = true;
        std::string msg = "\'" + std::string(p_variable.getNameCString()) + "\' declared as an array with an index that is not greater than 0";
        showErrorMessage(p_variable.getLocation().line, p_variable.getLocation().col, msg);
        cur_symbol_table->addError(p_variable.getNameCString());
    }
    
}

void SemanticAnalyzer::visit(ConstantValueNode &p_constant_value) {
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Travere child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */

    //ConstantValueNode does not have child nodes

    if(isVar){
        cur_symbol_entry->setKindString("constant");
        cur_symbol_entry->setAttrString(p_constant_value.getConstantValueCString());
    }
}

void SemanticAnalyzer::visit(FunctionNode &p_function) {
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Travere child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */

    //printf("FunctionNode\n");
    cur_symbol_table = symbol_manager->getTopSymbolTable();
    if(cur_symbol_table->checkRedecl(p_function.getNameCString())){
        isError = true;
        std::string msg = "symbol \'" + std::string(p_function.getNameCString()) + "\' is redeclared";
        showErrorMessage(p_function.getLocation().line, p_function.getLocation().col, msg);
    }
    else{
        cur_symbol_entry = new SymbolEntry(p_function.getNameCString(), "function", m_level, p_function.getPTypeCString(), p_function.getArgCString());
        cur_symbol_table->addSymbol(cur_symbol_entry);
    }
    cur_symbol_table = new SymbolTable();
    symbol_manager->pushScope(cur_symbol_table);
    m_level++;

    isFunc = true;
    isInFunc = true;
    cur_func_name = p_function.getNameCString();
    p_function.visitChildNodes(*this);
    cur_func_name = "";
    isFunc = false;
    isInFunc = false;

    if(opt_d){
        cur_symbol_table = symbol_manager->getTopSymbolTable();
        cur_symbol_table->dumpSymbolTable();
    }

    symbol_manager->popScope();
    m_level--;
}

void SemanticAnalyzer::visit(CompoundStatementNode &p_compound_statement) {
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Travere child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */

    bool to_pop = true;
    //Statement under a function
    if(isFunc){
        
        isFunc = false;
        to_pop = false;
        
    }
    else{
        //Not under a function
        cur_symbol_table = new SymbolTable();
        symbol_manager->pushScope(cur_symbol_table);
        m_level++;
    }

    p_compound_statement.visitChildNodes(*this);

    if(to_pop){
        if(opt_d){
            cur_symbol_table = symbol_manager->getTopSymbolTable();
            cur_symbol_table->dumpSymbolTable();
        }

        symbol_manager->popScope();
        m_level--;

    }

}

void SemanticAnalyzer::visit(PrintNode &p_print) {
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Travere child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */

    p_print.visitChildNodes(*this);
    if(p_print.checkInvalidChildren() == 1){
        return;
    }

    if(p_print.checkTargetScalarType() == 0){
        isError = true;
        std::string msg = "expression of print statement must be scalar type";
        showErrorMessage(p_print.getLocation().line, p_print.getTargetLocationCol(), msg);
    }
}

void SemanticAnalyzer::visit(BinaryOperatorNode &p_bin_op) {
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Travere child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */

    p_bin_op.visitChildNodes(*this);
    if(p_bin_op.checkInvalidChildren()){
        return;
    }
    std::string ltype = p_bin_op.getLTypeCString();
    std::string rtype = p_bin_op.getRTypeCString();
    std::string op = p_bin_op.getOpCString();
    if(op == "+" || op == "-" || op == "*" || op == "/"){
        if(op == "+" && ltype == "string" && rtype == "string"){
            p_bin_op.setNodeType("string");
        }
        else if(ltype == "integer" && rtype == "integer"){
            p_bin_op.setNodeType("integer");
        }
        else if(ltype == "real" && rtype == "real"){
            p_bin_op.setNodeType("real");
        }
        else if(ltype == "integer" && rtype == "real"){
            p_bin_op.setNodeType("real");
        }
        else if(ltype == "real" && rtype == "integer"){
            p_bin_op.setNodeType("real");
        }
        else{
            isError = true;
            std::string msg = "invalid operands to binary operator \'" + op + "\' (\'" + ltype + "\' and \'" + rtype + "\')";
            showErrorMessage(p_bin_op.getLocation().line, p_bin_op.getLocation().col, msg);
        }

    }
    else if(op == "mod"){
        if(ltype == "integer" && rtype == "integer"){
            p_bin_op.setNodeType("integer");
        }
        else{
            isError = true;
            std::string msg = "invalid operands to binary operator \'" + op + "\' (\'" + ltype + "\' and \'" + rtype + "\')";
            showErrorMessage(p_bin_op.getLocation().line, p_bin_op.getLocation().col, msg);
        }
    }
    else if(op == "and" || op == "or"){
        if(ltype == "boolean" && rtype == "boolean"){
            p_bin_op .setNodeType("boolean");
        }
        else{
            isError = true;
            std::string msg = "invalid operands to binary operator \'" + op + "\' (\'" + ltype + "\' and \'" + rtype + "\')";
            showErrorMessage(p_bin_op.getLocation().line, p_bin_op.getLocation().col, msg);
        }
    }
    //Comparison
    else{
        if(ltype == "integer" && rtype == "integer"){
            p_bin_op.setNodeType("boolean");
        }
        else if(ltype == "real" && rtype == "real"){
            p_bin_op.setNodeType("boolean");
        }
        else if(ltype == "integer" && rtype == "real"){
            p_bin_op.setNodeType("boolean");
        }
        else if(ltype == "real" && rtype == "integer"){
            p_bin_op.setNodeType("boolean");
        }
        else{
            isError = true;
            std::string msg = "invalid operands to binary operator \'" + op + "\' (\'" + ltype + "\' and \'" + rtype + "\')";
            showErrorMessage(p_bin_op.getLocation().line, p_bin_op.getLocation().col, msg);
        }
    }
}

//Checkpoint

void SemanticAnalyzer::visit(UnaryOperatorNode &p_un_op) {
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Travere child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */

    p_un_op.visitChildNodes(*this);
    if(p_un_op.checkInvalidChildren() == 1){
        return;
    }
    std::string op = p_un_op.getOpCString();
    std::string operand = p_un_op.getOperandTypeCString();
    if(op == "neg"){
        if(operand == "integer" || operand == "real"){
            p_un_op.setNodeType(operand);
        }
        else{
            isError = true;
            std::string msg = "invalid operand to unary operator \'" + op + "\' (\'" + operand + "\')";
            showErrorMessage(p_un_op.getLocation().line, p_un_op.getLocation().col, msg);
        }
    }
    else{
        if(operand == "boolean"){
            p_un_op.setNodeType(operand);
        }
        else{
            isError = true;
            std::string msg = "invalid operand to unary operator \'" + op + "\' (\'" + operand + "\')";
            showErrorMessage(p_un_op.getLocation().line, p_un_op.getLocation().col, msg);
        }
    }
}

void SemanticAnalyzer::visit(FunctionInvocationNode &p_func_invocation) {
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Travere child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */

    p_func_invocation.visitChildNodes(*this);

    while(symbol_manager->getScopeSize() != 0){
        cur_symbol_table = symbol_manager->getTopSymbolTable();
        symbol_manager->popScope();
        tmp_manager->pushScope(cur_symbol_table);

        if(cur_symbol_table->checkRedecl(p_func_invocation.getNameCString()) == 0 && symbol_manager->getScopeSize() != 0){
            continue;
        }
        else if(cur_symbol_table->checkRedecl(p_func_invocation.getNameCString()) == 0 && symbol_manager->getScopeSize() == 0){
            isError = true;
            std::string msg = "use of undeclared symbol \'" + std::string(p_func_invocation.getNameCString()) + "\'";
            showErrorMessage(p_func_invocation.getLocation().line, p_func_invocation.getLocation().col, msg);
            break;
        }

        cur_symbol_entry = cur_symbol_table->getSymbolEntry(p_func_invocation.getNameCString());
        if((std::string)cur_symbol_entry->getKindCString() != "function"){
            isError = true;
            std::string msg = "call of non-function symbol \'" + std::string(p_func_invocation.getNameCString()) + "\'";
            showErrorMessage(p_func_invocation.getLocation().line, p_func_invocation.getLocation().col, msg);
            break;
        }

        if(cur_symbol_entry->getFunctionParamNum() != p_func_invocation.getArgNum()){
            isError = true;
            std::string msg = "too few/much arguments provided for function \'" + std::string(p_func_invocation.getNameCString()) + "\'";
            showErrorMessage(p_func_invocation.getLocation().line, p_func_invocation.getLocation().col, msg);
            break;
        }

        if(p_func_invocation.CheckInvalidChildren() == 1){
            break;
        }

        checkArgRet* ret = p_func_invocation.checkArgType(cur_symbol_entry->getAttrCString());
        if(ret->wrongid != -1){
            isError = true;
            uint32_t w_col = p_func_invocation.getArgLocationCol(ret->wrongid);
            std::string msg = "incompatible type passing \'" + ret->arg + "\' to parameter of type \'" + ret->p + "\'";
            showErrorMessage(p_func_invocation.getLocation().line, w_col, msg);
            break;
        }

        std::string rtype = cur_symbol_entry->getTypeCString();
        p_func_invocation.setNodeType(rtype);
        break;
    }

    while(tmp_manager->getScopeSize() > 0){
        symbol_manager->pushScope(tmp_manager->getTopSymbolTable());
        tmp_manager->popScope();
    }
}

void SemanticAnalyzer::visit(VariableReferenceNode &p_variable_ref) {
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Travere child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */

    p_variable_ref.visitChildNodes(*this);
    if(p_variable_ref.checkInvalidChildren()){
        
        return;
    }
    cur_symbol_table = symbol_manager->getTopSymbolTable();
    if(cur_symbol_table->checkErrDecl(p_variable_ref.getNameCString())){
        //std::cout<<cur_symbol_table->checkErrDecl(p_variable_ref.getNameCString())<<std::endl;
        return;
    }

    while(symbol_manager->getScopeSize() != 0){
        cur_symbol_table = symbol_manager->getTopSymbolTable();
        symbol_manager->popScope();
        tmp_manager->pushScope(cur_symbol_table);

        
        if(cur_symbol_table->checkRedecl(p_variable_ref.getNameCString()) == 0 && symbol_manager->getScopeSize() != 0){
            continue;
        }

        
        else if(cur_symbol_table->checkRedecl(p_variable_ref.getNameCString()) == 0 && symbol_manager->getScopeSize() == 0){
            isError = true;
            std::string msg = "use of undeclared symbol \'" + std::string(p_variable_ref.getNameCString()) + "\'";
            showErrorMessage(p_variable_ref.getLocation().line, p_variable_ref.getLocation().col, msg);
            isError = false;
            break;
        }
        
        
        
        
        cur_symbol_entry = cur_symbol_table->getSymbolEntry(p_variable_ref.getNameCString());
        if((std::string)cur_symbol_entry->getKindCString() == "program" ||(std::string)cur_symbol_entry->getKindCString() == "function"){
            isError = true;
            std::string msg = "use of non-variable symbol \'" + std::string(p_variable_ref.getNameCString()) + "\'";
            showErrorMessage(p_variable_ref.getLocation().line, p_variable_ref.getLocation().col, msg);
            break;
        }

        
        
        
        
        int dim = p_variable_ref.getIndicesNum();
        if(dim > cur_symbol_entry->getTypeDim()){
            isError = true;
            std::string msg = "there is an over array subscript on \'" + std::string(p_variable_ref.getNameCString()) + "\'";
            showErrorMessage(p_variable_ref.getLocation().line, p_variable_ref.getLocation().col, msg);
            break;
        }
        
        int n_int_col = p_variable_ref.checkNonIntegerIndex();
        if(n_int_col != -1){
            isError = true;
            std::string msg = "index of array reference must be an integer";
            showErrorMessage(p_variable_ref.getLocation().line, (uint32_t)n_int_col, msg);
            break;
        }
        
        
        

        PTypeSharedPtr p_type;
        std::string primitive_type;
        std::string type_string;
        type_string = cur_symbol_entry->getTypeCString();
        if(type_string.find("[") != std::string::npos){
            if(type_string.substr(0, 4) == "void"){
                primitive_type = "void";
            }
            else if(type_string.substr(0, 7) == "integer"){
                primitive_type = "integer";
            }
            else if(type_string.substr(0, 4) == "real"){
                primitive_type = "real";
            }
            else if(type_string.substr(0, 7) == "boolean"){
                primitive_type = "boolean";
            }
            else if(type_string.substr(0, 6) == "string"){
                primitive_type = "string";
            }
        }
        else{
            primitive_type = type_string;
        }


        p_variable_ref.setNodeType(primitive_type);
        std::vector<uint64_t> dims;
        cur_symbol_entry->getNewTypeDims(dims, p_variable_ref.getIndicesNum());
        p_variable_ref.setNodeTypeDim(dims);
        break;
        
    }


    while(tmp_manager->getScopeSize() > 0){
        symbol_manager->pushScope(tmp_manager->getTopSymbolTable());
        tmp_manager->popScope();
    }

}

void SemanticAnalyzer::visit(AssignmentNode &p_assignment) {
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Travere child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */

    p_assignment.visitChildNodes(*this);
    if(p_assignment.checkInvalidLvalue() == 1){
        isloop_var = false;
        return;
    }

    if(p_assignment.checkLvalueScalarType() == 0){
        isError = true;
        std::string msg = "array assignment is not allowed";
        showErrorMessage(p_assignment.getLocation().line, p_assignment.getLvalueLocationCol(), msg);
        isloop_var = false;
        return;
    }

    bool check_Rval = true;
    while(symbol_manager->getScopeSize()!=0){
        cur_symbol_table = symbol_manager->getTopSymbolTable();
        symbol_manager->popScope();
        tmp_manager->pushScope(cur_symbol_table);

        if(cur_symbol_table->checkRedecl(p_assignment.getLvalueNameCString()) == 0){
            continue;
        }
        cur_symbol_entry = cur_symbol_table->getSymbolEntry(p_assignment.getLvalueNameCString());
        std::string lvalue_kind = cur_symbol_entry->getKindCString();
        if(lvalue_kind == "constant"){
            isError = true;
            std::string msg = "cannot assign to variable \'" + std::string(p_assignment.getLvalueNameCString()) + "\' which is a constant";
            showErrorMessage(p_assignment.getLocation().line, p_assignment.getLvalueLocationCol(), msg);
            check_Rval = false;
            break;
        }

        if(lvalue_kind == "loop_var" && !isloop_var){
            isError = true;
            std::string msg = "the value of loop variable cannot be modified inside the loop body";
            showErrorMessage(p_assignment.getLocation().line, p_assignment.getLvalueLocationCol(), msg);
            check_Rval = false;
            break;
        }
    }

    while(tmp_manager->getScopeSize() > 0){
        symbol_manager->pushScope(tmp_manager->getTopSymbolTable());
        tmp_manager->popScope();
    }

    if(p_assignment.checkInvalidRvalue()){
        isloop_var = false;
        return;
    }

    if(check_Rval){
        if(p_assignment.checkRvalueScalarType() == 0){
            isError = true;
            std::string msg = "array assignment is not allowed";
            showErrorMessage(p_assignment.getLocation().line, p_assignment.getRvalueLocationCol(), msg);
            
        }
        else if(p_assignment.checkCompatibleLRvalueType() == 0){
            isError = true;
            std::string msg;
            std::string lvalue_type = p_assignment.getLvalueTypeCString();
            std::string rvalue_type = p_assignment.getRvalueTypeCString();
            msg = "assigning to \'" + lvalue_type + "\' from incompatible type \'" + rvalue_type + "\'";
            showErrorMessage(p_assignment.getLocation().line, p_assignment.getLocation().col, msg);
        }
    }

    isloop_var = false;



}

void SemanticAnalyzer::visit(ReadNode &p_read) {
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Travere child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */

    p_read.visitChildNodes(*this);
    if(p_read.checkInvalidChildren() == 1){
        return;
    }
    if(p_read.checkTargetScalarType() == 0){
        isError = true;
        std::string msg = "variable reference of read statement must be scalar type";
        showErrorMessage(p_read.getLocation().line, p_read.getTargetLocationCol(), msg);
    }
    if(symbol_manager->checkConst(p_read.getTargetNameCString())){
        isError = true;
        std::string msg = "variable reference of read statement cannot be a constant or loop variable";
        showErrorMessage(p_read.getLocation().line, p_read.getTargetLocationCol(), msg);
    }
    if(symbol_manager->checkLoopVarRedecl(p_read.getTargetNameCString())){
        isError = true;
        std::string msg = "variable reference of read statement cannot be a constant or loop variable";
        showErrorMessage(p_read.getLocation().line, p_read.getTargetLocationCol(), msg);
    }
}

void SemanticAnalyzer::visit(IfNode &p_if) {
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Travere child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */

    p_if.visitChildNodes(*this);
    if(p_if.checkInvalidCondition() == 1){
        return;
    }

    if(p_if.checkConditionBoolType() == 0){
        isError = true;
        std::string msg = "the expression of condition must be boolean type";
        showErrorMessage(p_if.getLocation().line, p_if.getConditionLocationCol(), msg);
    }
}

void SemanticAnalyzer::visit(WhileNode &p_while) {
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Travere child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */

    p_while.visitChildNodes(*this);
    if(p_while.checkInvalidCondition() == 1){
        return;
    }

    if(p_while.checkConditionBoolType() == 0){
        isError = true;
        std::string msg = "the expression of condition must be boolean type";
        showErrorMessage(p_while.getLocation().line, p_while.getLocation().col, msg);
    }
}

void SemanticAnalyzer::visit(ForNode &p_for) {
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Travere child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */

    cur_symbol_table = new SymbolTable();
    symbol_manager->pushScope(cur_symbol_table);
    m_level++;
    isfor = true;
    isloop_var = true;
    p_for.visitChildNodes(*this);
    isfor = false;
    isloop_var = false;

    int ival = atoi(p_for.getInitValCString());
    int cval = atoi(p_for.getConditionValCString());
    if(ival > cval){
        isError = true;
        std::string msg = "the lower bound and upper bound of iteration count must be in the incremental order";
        showErrorMessage(p_for.getLocation().line, p_for.getLocation().col, msg);
    }

    symbol_manager->pop_loop_var();
    if(opt_d){
        cur_symbol_table = symbol_manager->getTopSymbolTable();
        cur_symbol_table->dumpSymbolTable();
    }
    symbol_manager->popScope();
    m_level--;

}

void SemanticAnalyzer::visit(ReturnNode &p_return) {
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Travere child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */

    p_return.visitChildNodes(*this);
    if(!isInFunc){
        isError = true;
        std::string msg = "program/procedure should not return a value";
        showErrorMessage(p_return.getLocation().line, p_return.getLocation().col, msg);
        return;
    }

    while(symbol_manager->getScopeSize() != 0){
        cur_symbol_table = symbol_manager->getTopSymbolTable();
        symbol_manager->popScope();
        tmp_manager->pushScope(cur_symbol_table);

        if(cur_symbol_table->checkRedecl(cur_func_name.c_str()) == 0){
            continue;
        }
        SymbolEntry* func_entry = cur_symbol_table->getSymbolEntry(cur_func_name.c_str());
        std::string ret_type = func_entry->getTypeCString();

        if(ret_type == "void"){
            isError = true;
            std::string msg = "program/procedure should not return a value";
            showErrorMessage(p_return.getLocation().line, p_return.getLocation().col, msg);
            break;
        }

        if(p_return.checkInvalidRetType() == 1){
            break;
        }

        std::string rtype = p_return.getRetTypeCString();
        if(ret_type != rtype){
            isError = true;
            std::string msg = "return \'" + rtype + "\' from a function with return type \'" + ret_type + "\'";
            showErrorMessage(p_return.getLocation().line, p_return.getRetLocationCol(), msg);
            break;
        }
    }

    while(tmp_manager->getScopeSize() > 0){
        symbol_manager->pushScope(tmp_manager->getTopSymbolTable());
        tmp_manager->popScope();
    }
}
