#include "sema/Error.hpp"
#include "sema/SemanticAnalyzer.hpp"
#include "visitor/AstNodeInclude.hpp"

extern int32_t opt_dump;

void dumpSymbol(SymbolTable *symbol_table){
    for (size_t i = 0; i < 110; i++){
        printf("=");
    }
    puts("");
    printf("%-33s%-11s%-11s%-17s%-11s\n", "Name", "Kind", "Level", "Type", "Attribute");
    for (size_t i = 0; i < 110; i++){
        printf("-");
    }
    puts("");
    symbol_table->PrintTable();
    for (size_t i = 0; i < 110; i++){
        printf("-");
    }
    puts("");
}

void SemanticAnalyzer::visit(ProgramNode &p_program){
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Traverse child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */
    SymbolTable *cur_table = new SymbolTable(symbolManager.getLevel());
	symbolManager.pushScope(cur_table);
    cur_table->addSymbol(p_program.getNameCString(), "program", cur_table->getLevel(), "void", p_program.getType());
    std::string p_str = p_program.getNameCString();
    returnTypeManager.func_in.push_back(p_str);
    symbolManager.cur_root.push_back("program");
	p_program.visitChildNodes(*this);
	symbolManager.cur_root.pop_back();
	returnTypeManager.func_in.pop_back();
    if(opt_dump){
        dumpSymbol(cur_table);
    }
    symbolManager.popScope();
}

void SemanticAnalyzer::visit(DeclNode &p_decl){
    p_decl.visitChildNodes(*this);
}

void SemanticAnalyzer::visit(VariableNode &p_variable){
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Traverse child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */
    SymbolTable *cur_table = symbolManager.getTableTop();
    if(symbolManager.RedeclarationError(p_variable.getNameCString())){
		m_error_printer.print(SymbolRedeclarationError(p_variable.getLocation(), p_variable.getNameCString()));
		isErrorBFlag = true;
		return;
	}
    bool falsch_dim = false;
    if(p_variable.DimensionError()){
        falsch_dim = true;
        m_error_printer.print(NonPositiveArrayDimensionError(p_variable.getLocation(), p_variable.getNameCString()));
    }
	if(symbolManager.cur_root.back() == "function"){
		cur_table->addSymbol(p_variable.getNameCString(), "parameter", cur_table->getLevel(), p_variable.getTypeCString(), falsch_dim, p_variable.getType());
		return;
	}
    else if(symbolManager.cur_root.back()=="for"){
		cur_table->addSymbol(p_variable.getNameCString(), "loop_var", cur_table->getLevel(), p_variable.getTypeCString(), falsch_dim, p_variable.getType());
		return;
	}
    else{
		cur_table->addSymbol(p_variable.getNameCString(), "variable", cur_table->getLevel(), p_variable.getTypeCString(), falsch_dim, p_variable.getType());
	}
	
	symbolManager.cur_root.push_back("variable");
	p_variable.visitChildNodes(*this);
    symbolManager.cur_root.pop_back();
}

void SemanticAnalyzer::visit(ConstantValueNode &p_constant_value){
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Traverse child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */
    p_constant_value.type_ptr = p_constant_value.getPType();
    if(symbolManager.cur_root.back() == "variable"){
		SymbolTable *cur_table = symbolManager.getTableTop();
		cur_table->addConstantEntry(p_constant_value.getConstantValueCString());
	}
    for(auto &r : symbolManager.cur_root){
		if(r == "variable_ref" && symbolManager.cur_root.back() != "binary"){
			if(p_constant_value.getType() != "integer"){
                if(!isConstantError){
                    m_error_printer.print(NonIntegerArrayIndexError(p_constant_value.getLocation()));
                    isReadError = true;
                    isPrintError = true;
                    isConstantError = true;
                }
            }
			return;
		}
	}
    if(symbolManager.cur_root.back() == "binary" || symbolManager.cur_root.back() == "unary"){
		contextManager.expr_type.push_back(&*p_constant_value.getTypeSharedPtr());
	}
    if(symbolManager.cur_root.back() == "function_invocation" && !contextManager.isParamError){
		std::string func_type = static_cast<std::string>(p_constant_value.getTypeSharedPtr()->getPTypeCString());
        PType *para_type = symbolManager.getParameterType(contextManager.func_name.back(), contextManager.param_num);
        std::string para_type_string = static_cast<std::string>(para_type->getPTypeCString());
        if(func_type != para_type_string){
            if(func_type != "integer" || std::string(para_type->getPTypeCString()) != "real"){
                m_error_printer.print(IncompatibleArgumentTypeError(p_constant_value.getLocation(), para_type, func_type));
                isPrintError = true;
                contextManager.isParamError = true;
                return;
            }
        }
        contextManager.param_num++;
	}

    if(symbolManager.cur_root.back() == "assignment"){
		assignmentTypeVector.push_back(&*p_constant_value.getTypeSharedPtr());
        if(symbolManager.cur_root[symbolManager.cur_root.size() - 2] == "for"){
            loopStart = std::stoi(p_constant_value.getConstantValueCString());
        }
    }
	if(symbolManager.cur_root.back() == "for"){
		loopEnd = std::stoi(p_constant_value.getConstantValueCString());
	}
}

void SemanticAnalyzer::visit(FunctionNode &p_function){
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Traverse child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */
    
    if(symbolManager.RedeclarationError(p_function.getNameCString())){
        m_error_printer.print(SymbolRedeclarationError(p_function.getLocation(), p_function.getNameCString()));
    }
    else{
        SymbolTable *cur_function_program = symbolManager.getTableTop();
        cur_function_program->addSymbol(p_function.getNameCString(),"function", cur_function_program->getLevel(),p_function.getTypeCString(), p_function.getParameterString(),p_function.getType(),p_function.getParametersType());
    }

    SymbolTable *cur_table = new SymbolTable(symbolManager.getLevel(),p_function.getNameCString());
	symbolManager.pushScope(cur_table);
    returnTypeManager.func_in.push_back(p_function.getNameCString());
	symbolManager.cur_root.push_back("function");
    p_function.visitChildNodes(*this);
	symbolManager.cur_root.pop_back();
	returnTypeManager.func_in.pop_back();
    if(opt_dump){
        dumpSymbol(cur_table);
    }
    symbolManager.popScope();
}

void SemanticAnalyzer::visit(CompoundStatementNode &p_compound_statement){
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Traverse child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */
    if(symbolManager.cur_root.back() == "function"){
		symbolManager.cur_root.push_back("compound");
		p_compound_statement.visitChildNodes(*this);
		symbolManager.cur_root.pop_back();
	}
    else{
        SymbolTable *cur_table = new SymbolTable(symbolManager.getLevel());
		symbolManager.pushScope(cur_table);
        symbolManager.cur_root.push_back("compound");
        p_compound_statement.visitChildNodes(*this);
		symbolManager.cur_root.pop_back();
        if(opt_dump){
            dumpSymbol(cur_table);
        }
        symbolManager.popScope();
    }
}

void SemanticAnalyzer::visit(PrintNode &p_print){
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Traverse child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */
    symbolManager.cur_root.push_back("print");
	p_print.visitChildNodes(*this);
	symbolManager.cur_root.pop_back();
    isPrintError = false;
}

void SemanticAnalyzer::visit(BinaryOperatorNode &p_bin_op){
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Traverse child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */
    symbolManager.cur_root.push_back("binary");
    p_bin_op.visitChildNodes(*this);
    symbolManager.cur_root.pop_back();
    PType *type_right_oper = contextManager.expr_type.back();
	contextManager.expr_type.pop_back();
    std::string op = p_bin_op.getOpCString();
	PType *type_left_oper = contextManager.expr_type.back();
	contextManager.expr_type.pop_back();
    if(static_cast<std::string>(type_right_oper->getPTypeCString()) == "error" || static_cast<std::string>(type_left_oper->getPTypeCString()) == "error"){
        return;
    }
    std::string type_left = static_cast<std::string>(type_left_oper->getPTypeCString());
    std::string type_right = static_cast<std::string>(type_right_oper->getPTypeCString());
    if(op == "+" || op == "-" || op == "*" || op == "/"){
        if(type_right == "integer"){
            if(type_left != "integer" && type_left != "real"){
                m_error_printer.print(InvalidBinaryOperandError(p_bin_op.getLocation(), p_bin_op.get_m_op(), type_left_oper, type_right_oper));
                contextManager.expr_type.push_back(new PType(PType::PrimitiveTypeEnum::kErrorType));
                isIfError = true;
				isWhileError = true;
            }
            else{
                p_bin_op.type_ptr = type_left_oper;
                if(symbolManager.cur_root.back()=="binary"){
                    contextManager.expr_type.push_back(type_left_oper);
                }
            }
        }
        else if(type_right == "real"){
            if(type_left != "integer" && type_left != "real"){
                m_error_printer.print(InvalidBinaryOperandError(p_bin_op.getLocation(), p_bin_op.get_m_op(), type_left_oper, type_right_oper));
                contextManager.expr_type.push_back(new PType(PType::PrimitiveTypeEnum::kErrorType));
                isIfError = true;
		        isWhileError = true;
            }
            else{
                p_bin_op.type_ptr = type_right_oper;
                if(symbolManager.cur_root.back() == "binary"){
					contextManager.expr_type.push_back(type_right_oper);
				}
            }
        }
        else if(type_left == "string" && op == "+" && type_right == "string"){
            p_bin_op.type_ptr = type_right_oper;
            if(symbolManager.cur_root.back()=="binary"){
                contextManager.expr_type.push_back(type_right_oper);
            }
        }
        else{
            m_error_printer.print(InvalidBinaryOperandError(p_bin_op.getLocation(), p_bin_op.get_m_op(), type_left_oper, type_right_oper));
            contextManager.expr_type.push_back(new PType(PType::PrimitiveTypeEnum::kErrorType));
            isIfError = true;
            isWhileError = true;
        }
    }
    else if(op == "mod"){
        if(type_left != "integer" || type_right != "integer"){
            m_error_printer.print(InvalidBinaryOperandError(p_bin_op.getLocation(), p_bin_op.get_m_op(), type_left_oper, type_right_oper));
            contextManager.expr_type.push_back(new PType(PType::PrimitiveTypeEnum::kErrorType));
            isIfError = true;
            isWhileError = true;
        }
        else{
            p_bin_op.type_ptr = type_right_oper;
            if(symbolManager.cur_root.back()=="binary"){
                contextManager.expr_type.push_back(type_right_oper);
            }
        }
    }
    else if(op == "<" || op == "<=" || op == "=" || op == ">=" || op == ">" || op == "<>"){
        if((type_left != "integer" && type_left != "real") || (type_right != "integer" && type_right != "real")){
            m_error_printer.print(InvalidBinaryOperandError(p_bin_op.getLocation(), p_bin_op.get_m_op(), type_left_oper, type_right_oper));
            contextManager.expr_type.push_back(new PType(PType::PrimitiveTypeEnum::kErrorType));
            isIfError=true;
            isWhileError=true;
            
        }
        else{
            p_bin_op.type_ptr = new PType(PType::PrimitiveTypeEnum::kBoolType);
            if(symbolManager.cur_root.back() == "binary"){
                contextManager.expr_type.push_back(p_bin_op.type_ptr);
            }
        }
    }
    else if(op == "and" || op == "or"){
        if(type_left != "boolean" || type_right != "boolean"){
            m_error_printer.print(InvalidBinaryOperandError(p_bin_op.getLocation(), p_bin_op.get_m_op(), type_left_oper, type_right_oper));
            contextManager.expr_type.push_back(new PType(PType::PrimitiveTypeEnum::kErrorType));
            isIfError=true;
            isWhileError=true;
        }
        else{
            p_bin_op.type_ptr = type_right_oper;
            if(symbolManager.cur_root.back() == "binary"){
                contextManager.expr_type.push_back(type_right_oper);
            }
        }
    }
}

void SemanticAnalyzer::visit(UnaryOperatorNode &p_un_op){
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Traverse child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */

    symbolManager.cur_root.push_back("unary");
	p_un_op.visitChildNodes(*this);
    symbolManager.cur_root.pop_back();

    std::string op = p_un_op.getOpCString();
    PType *operand_type = contextManager.expr_type.back();
	contextManager.expr_type.pop_back();
    std::string op_type = static_cast<std::string>(operand_type->getPTypeCString());

    if(op_type == "error"){
        return;
    }

    if(op == "not"){
        if(op_type != "boolean"){
            m_error_printer.print(InvalidUnaryOperandError(p_un_op.getLocation(), p_un_op.get_m_op(), operand_type));
            isIfError = true;
            isWhileError = true;
            if(symbolManager.cur_root.back() == "assignment"){ 
                assignmentTypeVector.push_back(new PType(PType::PrimitiveTypeEnum::kErrorType));
            }
        }
        else{
            p_un_op.type_ptr = operand_type;
			if(symbolManager.cur_root.back() == "unary"){
				contextManager.expr_type.push_back(operand_type);
			}
            if(symbolManager.cur_root.back() == "assignment"){ 
                assignmentTypeVector.push_back(operand_type);
            }
        }
    }
    else if(op == "-" || op == "neg"){
        if(op_type != "real" && op_type != "integer"){
            m_error_printer.print(InvalidUnaryOperandError(p_un_op.getLocation(), p_un_op.get_m_op(), operand_type));
            isIfError = true;
            isWhileError = true;
            if(symbolManager.cur_root.back() == "function_invocation"){
                contextManager.isParamError = true;
                isPrintError = true;
            }
            if(symbolManager.cur_root.back() == "assignment"){ 
                assignmentTypeVector.push_back(new PType(PType::PrimitiveTypeEnum::kErrorType));
            }
        }
        else{
            p_un_op.type_ptr = operand_type;
			if(symbolManager.cur_root.back() == "unary"){
				contextManager.expr_type.push_back(operand_type);
			}
            if(symbolManager.cur_root.back() == "assignment"){ 
                assignmentTypeVector.push_back(operand_type);
            }
        }
    }
}

void SemanticAnalyzer::visit(FunctionInvocationNode &p_func_invocation){
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Traverse child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */
    if(!symbolManager.UndeclaredError(p_func_invocation.getNameCString())){
        m_error_printer.print(UndeclaredSymbolError(p_func_invocation.getLocation(), p_func_invocation.getNameCString()));
		return;
    }
    if(!symbolManager.NonFunctionSymbolError(p_func_invocation.getNameCString())){
        m_error_printer.print(NonFunctionSymbolError(p_func_invocation.getLocation(), p_func_invocation.getNameCString()));
		return;
    }
    if(symbolManager.ArgumentNumberMismatchError(p_func_invocation.getNameCString(), p_func_invocation.getParaSize())){
        m_error_printer.print(ArgumentNumberMismatchError(p_func_invocation.getLocation(), p_func_invocation.getNameCString()));
        return;
    }

    p_func_invocation.type_ptr = symbolManager.getFunctionType(p_func_invocation.getNameCString());
    if(symbolManager.cur_root.back() == "function_invocation" && !contextManager.isParamError){
        std::string func_type = static_cast<std::string>(symbolManager.getFunctionType(p_func_invocation.getNameCString())->getPTypeCString());
        PType *para_type = symbolManager.getParameterType(contextManager.func_name.back(), contextManager.param_num);
        std::string para_type_string = static_cast<std::string>(para_type->getPTypeCString());
        if(func_type != para_type_string){
            if(func_type != "integer" || para_type_string != "real"){
                m_error_printer.print(IncompatibleArgumentTypeError(p_func_invocation.getLocation(), para_type, func_type));
                contextManager.isParamError = true;
                return;
            }
        }
        contextManager.param_num = contextManager.param_num + 1;
    }

    if(symbolManager.cur_root.back() == "assignment"){
		assignmentTypeVector.push_back(symbolManager.getType(p_func_invocation.getNameCString()));
	}
    if(symbolManager.cur_root.back() == "binary" || symbolManager.cur_root.back() == "unary"){
		contextManager.expr_type.push_back(symbolManager.getType(p_func_invocation.getNameCString()));
	}

	symbolManager.cur_root.push_back("function_invocation");
    contextManager.func_name.push_back(p_func_invocation.getNameCString());
	p_func_invocation.visitChildNodes(*this);
	symbolManager.cur_root.pop_back();
	contextManager.func_name.pop_back();
    
	contextManager.isParamError = false;
    contextManager.param_num = 0;

    if(symbolManager.cur_root.back() == "print"){
		if(!isPrintError){
            if(p_func_invocation.type_ptr->getType() == "void"){
                m_error_printer.print(PrintOutNonScalarTypeError(p_func_invocation.getLocation()));
                isPrintError = true;
            }
        }
	}
}

void SemanticAnalyzer::visit(VariableReferenceNode &p_variable_ref){
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Traverse child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */
    
    if(!symbolManager.UndeclaredError(p_variable_ref.getNameCString())){
        m_error_printer.print(UndeclaredSymbolError(p_variable_ref.getLocation(), p_variable_ref.getNameCString()));
        isErrorBFlag = true;
		return;
    }
    if(!symbolManager.NonVariableError(p_variable_ref.getNameCString())){
        m_error_printer.print(NonVariableSymbolError(p_variable_ref.getLocation(), p_variable_ref.getNameCString()));
        isErrorBFlag = true;
		return;
    }
    if(symbolManager.WrongDeclOrNot(p_variable_ref.getNameCString())){ 
        return; 
    }
    isConstantError = false;
    p_variable_ref.type_ptr=symbolManager.getType(p_variable_ref.getNameCString());
	symbolManager.cur_root.push_back("variable_ref");

    p_variable_ref.visitChildNodes(*this);
	symbolManager.cur_root.pop_back();
    
    if(symbolManager.OverArraySubscriptError(p_variable_ref.getNameCString(), p_variable_ref.getSizeOfDimension())){
        if(isConstantError)
            return;
        m_error_printer.print(OverArraySubscriptError(p_variable_ref.getLocation(), p_variable_ref.getNameCString()));
        contextManager.expr_type.push_back(new PType(PType::PrimitiveTypeEnum::kErrorType));
        isErrorBFlag = true;
        isPrintError = true;
        isReadError = true;
    }

    if(symbolManager.cur_root.back()=="binary" || symbolManager.cur_root.back() == "unary"){
		PType *cur_type = new PType(symbolManager.getType(p_variable_ref.getNameCString())->getPrimitiveType());
		std::vector<uint64_t> cur_dim;
        int size_type_dim = symbolManager.getType(p_variable_ref.getNameCString())->getDimensions().size();
		for(int i = 0; i < size_type_dim; i++){
			if(i >= p_variable_ref.getSizeOfDimension()){
				cur_dim.push_back(symbolManager.getType(p_variable_ref.getNameCString())->getDimensions()[i]);
			}
		}
		cur_type->setDimensions(cur_dim);
		contextManager.expr_type.push_back(cur_type);
	}
    if(symbolManager.cur_root.back() == "function_invocation" && !contextManager.isParamError ){
		std::string func_type = static_cast<std::string>(symbolManager.getType(p_variable_ref.getNameCString())->getPTypeCString());
        PType *para_type = symbolManager.getParameterType(contextManager.func_name.back(), contextManager.param_num);
        std::string para_type_string = static_cast<std::string>(para_type->getPTypeCString());
        if(func_type != para_type_string){
            if(func_type != "integer" || para_type_string != "real"){
                m_error_printer.print(IncompatibleArgumentTypeError(p_variable_ref.getLocation(), para_type, func_type));
                isPrintError = true;
                isReadError = true;
                contextManager.isParamError = true;
                return;
            }
        }
        contextManager.param_num++;
	}
    if(symbolManager.cur_root.back() == "print"){
		if(!isPrintError){
            int type_dim = p_variable_ref.type_ptr->getDimensionsSize();
            int actual_dim = p_variable_ref.getSizeOfDimension();
            if(p_variable_ref.type_ptr->getType() == "void" || actual_dim < type_dim){
                m_error_printer.print(PrintOutNonScalarTypeError(p_variable_ref.getLocation()));
                isPrintError = true;
            }
        }
	}
    if(symbolManager.cur_root.back() == "read"){
        if(!isReadError && !isConstantError){
            int type_dim = p_variable_ref.type_ptr->getDimensionsSize();
            int actual_dim = p_variable_ref.getSizeOfDimension();
            bool isLoopVar = symbolManager.LoopVarOrNot(p_variable_ref.getNameCString());
            std::string k_str = symbolManager.getKind(p_variable_ref.getNameCString());
            if(p_variable_ref.type_ptr->getType() == "void" || actual_dim < type_dim){
                m_error_printer.print(ReadToNonScalarTypeError(p_variable_ref.getLocation()));
                isReadError = true;
            }
            else if(isLoopVar || k_str == "constant"){
                m_error_printer.print(ReadToConstantOrLoopVarError(p_variable_ref.getLocation()));
                isReadError = true;
            }
        }
    }
    if(symbolManager.cur_root.back() == "assignment"){
		PType *cur_type = new PType(symbolManager.getType(p_variable_ref.getNameCString())->getPrimitiveType());
		std::vector<uint64_t> cur_dim;
		for(int i = 0; i < symbolManager.getType(p_variable_ref.getNameCString())->getDimensions().size(); i++){
			if(i >= p_variable_ref.getSizeOfDimension()){
				cur_dim.push_back(symbolManager.getType(p_variable_ref.getNameCString())->getDimensions()[i]);
			}
		}
		cur_type->setDimensions(cur_dim);
		assignmentTypeVector.push_back(cur_type);
	}
}

void SemanticAnalyzer::visit(AssignmentNode &p_assignment){
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Traverse child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */
    symbolManager.cur_root.push_back("assignment");
	p_assignment.visitChildNodes(*this);
	symbolManager.cur_root.pop_back();
    VariableReferenceNode *p_var_ref = p_assignment.getVarNode();
    ExpressionNode *p_expr_node = p_assignment.getExprNode();
    std::string var_name = p_var_ref->getNameCString();

    if(!isErrorBFlag){
        
        if(p_assignment.getVarDim() > 0){
            m_error_printer.print(AssignWithArrayTypeError(p_var_ref->getLocation()));
            isErrorBFlag = true;
        }
        else if(symbolManager.getKind(var_name) == "constant"){
            m_error_printer.print(AssignToConstantError(p_var_ref->getLocation(), p_var_ref->getNameCString()));
            isErrorBFlag = true;
        }
        else if(symbolManager.cur_root.back() != "for" && symbolManager.getKind(var_name) == "loop_var"){
            m_error_printer.print(AssignToLoopVarError(p_var_ref->getLocation()));
            isErrorBFlag = true;
        }
        else if(p_assignment.getExprDim() > 0){
            m_error_printer.print(AssignWithArrayTypeError(p_expr_node->getLocation()));
            isErrorBFlag = true;
        }
            
    }

    if(isErrorBFlag){
		assignmentTypeVector.clear();
        isErrorBFlag = false;
		return;
    }

    PType *r = assignmentTypeVector.back();
    assignmentTypeVector.pop_back();
    PType *l = assignmentTypeVector.back();
    assignmentTypeVector.pop_back();
    std::string r_str = static_cast<std::string>(r->getPTypeCString());
    std::string l_str = static_cast<std::string>(l->getPTypeCString());
    if(r_str == "error" || l_str == "error"){
        return;
    }
    else if(r_str != l_str && (r_str != "integer" || l_str != "real")){
        m_error_printer.print(IncompatibleAssignmentError(p_assignment.getLocation(), l, r));
    }
    
}

void SemanticAnalyzer::visit(ReadNode &p_read){
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Traverse child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */
    symbolManager.cur_root.push_back("read");
    p_read.visitChildNodes(*this);
    isReadError = false;
    symbolManager.cur_root.pop_back();
}

void SemanticAnalyzer::visit(IfNode &p_if){
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Traverse child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */
    symbolManager.cur_root.push_back("if");
    p_if.visitChildNodes(*this);
    symbolManager.cur_root.pop_back();
    std::string cond_type = static_cast<std::string>(p_if.getConditionType());
	if(cond_type != "boolean"){
		if(!isIfError){
            m_error_printer.print(NonBooleanConditionError(p_if.getCondition()->getLocation()));
        }
	}
	isIfError = false;
}

void SemanticAnalyzer::visit(WhileNode &p_while){
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Traverse child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */
    symbolManager.cur_root.push_back("while");
	p_while.visitChildNodes(*this);
	symbolManager.cur_root.pop_back();
    std::string condition_type = static_cast<std::string>(p_while.getConditionType());
    if(condition_type!= "boolean"){
		if(!isWhileError){
            m_error_printer.print(NonBooleanConditionError(p_while.getCondition()->getLocation()));
        }
	}
	isWhileError = false;
}

void SemanticAnalyzer::visit(ForNode &p_for){
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Traverse child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */
    SymbolTable *cur_table = new SymbolTable(symbolManager.getLevel());
	symbolManager.pushScope(cur_table);
	symbolManager.cur_root.push_back("for");
    p_for.visitChildNodes(*this);
	symbolManager.cur_root.pop_back();
    if(isForError()){
        m_error_printer.print(NonIncrementalLoopVariableError(p_for.getLocation()));
    }
    if(opt_dump){
        dumpSymbol(cur_table);
    }
    symbolManager.popScope();
}

void SemanticAnalyzer::visit(ReturnNode &p_return){
    /*
     * TODO:
     *
     * 1. Push a new symbol table if this node forms a scope.
     * 2. Insert the symbol into current symbol table if this node is related to
     *    declaration (ProgramNode, VariableNode, FunctionNode).
     * 3. Traverse child nodes of this node.
     * 4. Perform semantic analyses of this node.
     * 5. Pop the symbol table pushed at the 1st step.
     */
    symbolManager.cur_root.push_back("return");
	p_return.visitChildNodes(*this);
    std::string enter_func_name = returnTypeManager.func_in.back();
    if(returnTypeManager.func_in.size() == 1 || symbolManager.getFunctionType(enter_func_name)->getType() == "void"){
        m_error_printer.print(ReturnFromVoidError(p_return.getLocation()));
        return;
    }
    ExpressionNode *ret_expr = p_return.get_m_ret_val();
    std::string expr_type = returnTypeManager.getReturnTypeString(ret_expr->getSizeOfDimension(), ret_expr->type_ptr);
    PType *func_type = symbolManager.getFunctionType(enter_func_name);

    std::string func_type_string = static_cast<std::string>(func_type->getPTypeCString());
    if(expr_type != func_type_string && expr_type != "error"){
		if(func_type_string != "real" || expr_type != "integer"){
            m_error_printer.print(IncompatibleReturnTypeError(ret_expr->getLocation(), func_type, expr_type));
        }
	}
    symbolManager.cur_root.pop_back();
}