%{
#include "AST/BinaryOperator.hpp"
#include "AST/CompoundStatement.hpp"
#include "AST/ConstantValue.hpp"
#include "AST/FunctionInvocation.hpp"
#include "AST/UnaryOperator.hpp"
#include "AST/VariableReference.hpp"
#include "AST/assignment.hpp"
#include "AST/ast.hpp"
#include "AST/decl.hpp"
#include "AST/expression.hpp"
#include "AST/for.hpp"
#include "AST/function.hpp"
#include "AST/if.hpp"
#include "AST/print.hpp"
#include "AST/program.hpp"
#include "AST/read.hpp"
#include "AST/return.hpp"
#include "AST/variable.hpp"
#include "AST/while.hpp"
#include "AST/AstDumper.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define YYLTYPE yyltype

typedef struct YYLTYPE {
    uint32_t first_line;
    uint32_t first_column;
    uint32_t last_line;
    uint32_t last_column;
} yyltype;

extern uint32_t line_num;   /* declared in scanner.l */
extern char current_line[]; /* declared in scanner.l */
extern FILE *yyin;          /* declared by lex */
extern char *yytext;        /* declared by lex */

static AstNode *root;

extern "C" int yylex(void);
static void yyerror(const char *msg);
extern int yylex_destroy(void);
%}

// This guarantees that headers do not conflict when included together.
%define api.token.prefix {TOK_}

%code requires {
    #include <vector>
    #include "AST/ast.hpp"
    class AstNode; 
    class CompoundStatementNode; 
    class FunctionInvocationNode; 
    class VariableReferenceNode; 
    class AssignmentNode; 
    class ConstantValueNode; 
    class BinaryOperatorNode; 
    class UnaryOperatorNode;
    class PrintNode; 
    class ReadNode; 
    class IfNode; 
    class WhileNode; 
    class ForNode; 
    class ReturnNode; 
    class ProgramNode; 
    class FunctionNode; 
    class VariableNode; 
    class DeclNode; 
    class ExpressionNode; 

    struct Variable {
        std::string type, val;
        size_t line, column;

        Variable(std::string type, std::string val) : type(type), val(val) {}
        Variable(std::string val, size_t line, size_t column) : type("ID"), val(val), line(line), column(column) {}
    };
}

    /* For yylval */
%union {
    /* basic semantic value */
    char *identifier;
    int integer;
    double real;
    char *string;

    Variable *variable;



    AstNode *node;
    CompoundStatementNode *compound_stmt_ptr;
    FunctionInvocationNode *function_invocation_ptr;
    VariableReferenceNode *variable_reference_ptr;
    AssignmentNode *assignment_ptr;
    ConstantValueNode *constant_value_ptr;
    BinaryOperatorNode *binary_operator_ptr;
    UnaryOperatorNode *unary_operator_ptr;
    PrintNode *print_ptr;
    ReadNode *read_ptr;
    IfNode *if_ptr;
    WhileNode *while_ptr;
    ForNode *for_ptr;
    ReturnNode *return_ptr;
    ProgramNode *program_ptr;
    FunctionNode *function_ptr;
    VariableNode *variable_ptr;
    DeclNode *decl_ptr;
    ExpressionNode *expression_ptr;

    std::vector<DeclNode *> *decl_list_ptr;
    std::vector<FunctionNode *> *function_list_ptr;
    std::vector<AstNode *> *node_list_ptr;
    std::vector<Variable *> *variable_list_ptr;
    std::vector<ExpressionNode *> *expression_list_ptr;



};

%type <identifier> ProgramName ID Type ScalarType IntegerAndReal ArrType ArrDecl STRING_LITERAL FunctionName ReturnType
%type <integer> INT_LITERAL NegOrNot
%type <compound_stmt_ptr> CompoundStatement ElseOrNot
%type <program_ptr> Program
%type <function_ptr> Function FunctionDeclaration FunctionDefinition
%type <function_list_ptr> Functions FunctionList
%type <decl_ptr> Declaration FormalArg
%type <decl_list_ptr> Declarations DeclarationList FormalArgs FormalArgList
%type <variable> LiteralConstant StringAndBoolean
%type <variable_list_ptr> IdList
%type <real> REAL_LITERAL
%type <node_list_ptr> StatementList Statements
%type <node> ProgramUnit Statement Simple
%type <variable_reference_ptr> VariableReference
%type <expression_list_ptr> ExpressionList Expressions ArrRefList ArrRefs
%type <expression_ptr> Expression
%type <if_ptr> Condition
%type <while_ptr> While
%type <for_ptr> For
%type <return_ptr> Return
%type <function_invocation_ptr> FunctionCall FunctionInvocation


    /* Follow the order in scanner.l */

    /* Delimiter */
%token COMMA SEMICOLON COLON
%token L_PARENTHESIS R_PARENTHESIS
%token L_BRACKET R_BRACKET

    /* Operator */
%token ASSIGN
    /* TODO: specify the precedence of the following operators */
%left OR
%left AND
%left EQUAL NOT_EQUAL
%left LESS GREATER LESS_OR_EQUAL GREATER_OR_EQUAL
%left PLUS MINUS
%left MULTIPLY DIVIDE MOD
%right NOT UNARY_MINUS


    /* Keyword */
%token ARRAY BOOLEAN INTEGER REAL STRING
%token END BEGIN
%token DO ELSE FOR IF THEN WHILE
%token DEF OF TO RETURN VAR
%token FALSE TRUE
%token PRINT READ

    /* Identifier */
%token ID

    /* Literal */
%token INT_LITERAL
%token REAL_LITERAL
%token STRING_LITERAL

%%

ProgramUnit:
    Program{
        $$ = $1;
    }
    |
    Function{
        $$ = $1;
    }
;

Program:
    ProgramName SEMICOLON
    /* ProgramBody */
    DeclarationList FunctionList CompoundStatement
    /* End of ProgramBody */
    END {
        root = new ProgramNode(@1.first_line, @1.first_column,
                               $1, $3, $4, $5);

        free($1);
    }
;

ProgramName:
    ID
;

DeclarationList:
    Epsilon{
        $$ = NULL;
    }
    |
    Declarations{
        $$ = $1;
    }
;

Declarations:
    Declaration{
        $$ = new std::vector<DeclNode *>();
        $$->push_back($1);
    }
    |
    Declarations Declaration{
        $$ = $1;
        $$->push_back($2);
    }
;

FunctionList:
    Epsilon{
        $$ = NULL;
    }
    |
    Functions{
        $$ = $1;
    }
;

Functions:
    Function{
        $$ = new std::vector<FunctionNode *>();
        $$->push_back($1);
    }
    |
    Functions Function{
        $$ = $1;
        $$->push_back($2);
    }
;

Function:
    FunctionDeclaration{
        $$ = $1;
    }
    |
    FunctionDefinition{
        $$ = $1;
    }
;

FunctionDeclaration:
    FunctionName L_PARENTHESIS FormalArgList R_PARENTHESIS ReturnType SEMICOLON{
        if ($3 == NULL) {
            $$ = new FunctionNode(@1.first_line, @1.first_column, $1, NULL, $5, NULL);
        } else {
            std::vector<DeclNode*> *decl_list = new std::vector<DeclNode*>();
            for (auto arg : *$3) {
                DeclNode *decl_formal_arg = (DeclNode *)arg;
                decl_list->push_back(decl_formal_arg);
            }
            $$ = new FunctionNode(@1.first_line, @1.first_column, $1, decl_list, $5, NULL);
        }
    }
;

FunctionDefinition:
    FunctionName L_PARENTHESIS FormalArgList R_PARENTHESIS ReturnType
    CompoundStatement
    END{
        if ($3 == NULL) {
            $$ = new FunctionNode(@1.first_line, @1.first_column, $1, NULL, $5, $6);
        } else {
            std::vector<DeclNode*>* decl_list = new std::vector<DeclNode*>();
            for (auto arg : *$3) {
                DeclNode *decl_formal_arg = (DeclNode *)arg;
                decl_list->push_back(decl_formal_arg);
            }
            $$ = new FunctionNode(@1.first_line, @1.first_column, $1, decl_list, $5, $6);
        }
    }
;

FunctionName:
    ID
;

FormalArgList:
    Epsilon{
        $$ = NULL;
    }
    |
    FormalArgs{
        $$ = $1;
    }
;

FormalArgs:
    FormalArg{
        $$ = new std::vector<DeclNode *>();
        $$->push_back($1);
    }
    |
    FormalArgs SEMICOLON FormalArg{
        $$ = $1;
        $$->push_back($3);
    }
;

FormalArg:
    IdList COLON Type{
        std::vector<VariableNode *> *var_list = new std::vector<VariableNode *>();
        for (auto id : *$1) {
            var_list->push_back(new VariableNode(id->line, id->column, id->val.c_str(), $3, NULL));
        }
        $$ = new DeclNode(@1.first_line, @1.first_column, var_list);
    }
;

IdList:
    ID{
        $$ = new std::vector<Variable *>();
        Variable *var = new Variable($1, @1.first_line, @1.first_column);
        $$->push_back(var);
    }
    |
    IdList COMMA ID{
        $$ = $1;
        Variable *var = new Variable($3, @3.first_line, @3.first_column);
        $$->push_back(var);
    }
;

ReturnType:
    COLON ScalarType{
        $$ = (char*)$2;
    }
    |
    Epsilon{
        $$ = (char*)"void";
    }
;

    /*
       Data Types and Declarations
                                   */

Declaration:
    VAR IdList COLON Type SEMICOLON{
        std::vector<VariableNode *> *id_list = new std::vector<VariableNode *>();
        for (auto id : *$2) {
            VariableNode *var = new VariableNode(id->line, id->column, id->val.c_str(), $4, NULL);
            id_list->push_back(var);
        }
        $$ = new DeclNode(@1.first_line, @1.first_column, id_list);
    }
    |
    VAR IdList COLON LiteralConstant SEMICOLON{
        std::vector<VariableNode *> *id_list = new std::vector<VariableNode *>();
        ConstantValueNode *constant_val = new ConstantValueNode(@1.first_line, $4->column, $4->val.c_str());
        for (auto id : *$2) {
            VariableNode *var = new VariableNode(id->line, id->column, id->val.c_str(), $4->type.c_str(), constant_val);
            id_list->push_back(var);
        }
        $$ = new DeclNode(@1.first_line, @1.first_column, id_list);
    }
;

Type:
    ScalarType{
        $$ = $1;
    }
    |
    ArrType{
        $$ = $1;
    }
;

ScalarType:
    INTEGER{
        $$ = (char *)"integer";
    }
    |
    REAL{
        $$ = (char *)"real";
    }
    |
    STRING{
        $$ = (char *)"string";
    }
    |
    BOOLEAN{
        $$ = (char *)"boolean";
    }
;

ArrType:
    ArrDecl ScalarType{
        std::string arrdecl = std::string($1);
        std::string type = std::string($2);
        std::string tmp = type + " " + arrdecl;
        $$ = strdup(tmp.c_str());

    }
;

ArrDecl:
    ARRAY INT_LITERAL OF{
        std::string int_literal = std::to_string($2);
        std::string tmp = "[" + int_literal + "]";
        $$ = strdup(tmp.c_str());
    }
    |
    ArrDecl ARRAY INT_LITERAL OF{
        std::string int_literal = std::to_string($3);
        std::string arrdecl = std::string($1);
        std::string tmp = arrdecl + "[" + int_literal + "]";
        $$ = strdup(tmp.c_str());
    }
;

LiteralConstant:
    NegOrNot INT_LITERAL{
        if ($1 == 1){
            std::string val = std::to_string($2);
            $$ = new Variable("integer", val);
            $$->column = @2.first_column;
        }
        else{
            int val = (int)$2;
            val = -val;
            std::string val_ = std::to_string(val);
            $$ = new Variable("integer", val_);
            $$->column = @1.first_column;
        }
    }
    |
    NegOrNot REAL_LITERAL{
        if ($1 == 1){
            std::string val = std::to_string($2);
            $$ = new Variable("real", val);
            $$->column = @2.first_column;
        }
        else{
            double val = (double)$2;
            val = -val;
            std::string val_ = std::to_string(val);
            $$ = new Variable("real", val_);
            $$->column = @1.first_column;
        }
    }
    |
    StringAndBoolean{
        $$ = $1;
    }
;

NegOrNot:
    Epsilon{
        $$ = 1;
    }
    |
    MINUS %prec UNARY_MINUS{
        $$ = -1;
    }
;

StringAndBoolean:
    STRING_LITERAL{
        $$ = new Variable("string", $1);
        $$->column = @1.first_column;
    }
    |
    TRUE{
        $$ = new Variable("boolean", "true");
        $$->column = @1.first_column;
    }
    |
    FALSE{
        $$ = new Variable("boolean", "false");
        $$->column = @1.first_column;
    }
;

IntegerAndReal:
    INT_LITERAL{
        std::string val = std::to_string($1);
        $$ = strdup(val.c_str());
    }
    |
    REAL_LITERAL{
        std::string val = std::to_string($1);
        $$ = strdup(val.c_str());
    }
;

    /*
       Statements
                  */

Statement:
    CompoundStatement{
        $$ = $1;
    }
    |
    Simple{
        $$ = $1;
    }
    |
    Condition{
        $$ = $1;
    }
    |
    While{
        $$ = $1;
    }
    |
    For{
        $$ = $1;
    }
    |
    Return{
        $$ = $1;
    }
    |
    FunctionCall{
        $$ = $1;
    }
;

CompoundStatement:
    BEGIN
    DeclarationList
    StatementList
    END{
        $$ = new CompoundStatementNode(@1.first_line, @1.first_column, $2, $3);
    }
;

Simple:
    VariableReference ASSIGN Expression SEMICOLON{
        $$ = new AssignmentNode(@2.first_line, @2.first_column, $1, $3);
    }
    |
    PRINT Expression SEMICOLON{
        $$ = new PrintNode(@1.first_line, @1.first_column, $2);
    }
    |
    READ VariableReference SEMICOLON{
        $$ = new ReadNode(@1.first_line, @1.first_column, $2);
    }
;

VariableReference:
    ID ArrRefList{
        $$ = new VariableReferenceNode(@1.first_line, @1.first_column, $1, $2);
        //free($1);
    }
;

ArrRefList:
    Epsilon{
        $$ = NULL;
    }
    |
    ArrRefs{
        $$ = $1;
    }
;

ArrRefs:
    L_BRACKET Expression R_BRACKET{
        $$ = new std::vector<ExpressionNode *>();
        $$->push_back($2);
    }
    |
    ArrRefs L_BRACKET Expression R_BRACKET{
        $$ = $1;
        $$->push_back($3);
    }
;

Condition:
    IF Expression THEN
    CompoundStatement
    ElseOrNot
    END IF{
        $$ = new IfNode(@1.first_line, @1.first_column, $2, $4, $5);
    }
;

ElseOrNot:
    ELSE
    CompoundStatement{
        $$ = $2;
    }
    |
    Epsilon{
        $$ = NULL;
    }
;

While:
    WHILE Expression DO
    CompoundStatement
    END DO{
        $$ = new WhileNode(@1.first_line, @1.first_column, $2, $4);
    }
;

For:
    FOR ID ASSIGN INT_LITERAL TO INT_LITERAL DO
    CompoundStatement
    END DO{
        VariableNode *var = new VariableNode(@2.first_line, @2.first_column, $2, "integer", NULL);
        std::vector<VariableNode *> *var_list = new std::vector<VariableNode *>({var});
        VariableReferenceNode *var_ref = new VariableReferenceNode(@1.first_line, @2.first_column, $2, NULL);
        ConstantValueNode *assign_const = new ConstantValueNode(@1.first_line, @4.first_column, std::to_string($4).c_str());
        DeclNode *decl = new DeclNode(@1.first_line, @2.first_column, var_list);
        AssignmentNode *assign = new AssignmentNode(@1.first_line, @3.first_column, var_ref, assign_const);
        ExpressionNode *expr = new ConstantValueNode(@1.first_line, @6.first_column, std::to_string($6).c_str());
    
        $$ = new ForNode(@1.first_line, @1.first_column, decl, assign, expr, $8);
    }
;

Return:
    RETURN Expression SEMICOLON{
        $$ = new ReturnNode(@1.first_line, @1.first_column, $2);
    }
;

FunctionCall:
    FunctionInvocation SEMICOLON{
        $$ = $1;
    }
;

FunctionInvocation:
    ID L_PARENTHESIS ExpressionList R_PARENTHESIS{
        $$ = new FunctionInvocationNode(@1.first_line, @1.first_column, $1, $3);
    }
;

ExpressionList:
    Epsilon{
        $$ = NULL;
    }
    |
    Expressions{
        $$ = $1;
    }
;

Expressions:
    Expression{
        $$ = new std::vector<ExpressionNode *>();
        $$->push_back($1);
    }
    |
    Expressions COMMA Expression{
        $$ = $1;
        $$->push_back($3);
    }
;

StatementList:
    Epsilon{
        $$ = NULL;
    }
    |
    Statements{
        $$ = $1;
    }
;

Statements:
    Statement{
        $$ = new std::vector<AstNode *>();
        $$->push_back($1);
    }
    |
    Statements Statement{
        $$ = $1;
        $$->push_back($2);
    }
;

Expression:
    L_PARENTHESIS Expression R_PARENTHESIS{
        $$ = $2;
    }
    |
    MINUS Expression %prec UNARY_MINUS{
        $$ = new UnaryOperatorNode(@1.first_line, @1.first_column, "neg", $2);
    }
    |
    Expression MULTIPLY Expression{
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, "*", $1, $3);
    }
    |
    Expression DIVIDE Expression{
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, "/", $1, $3);
    }
    |
    Expression MOD Expression{
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, "mod", $1, $3);
    }
    |
    Expression PLUS Expression{
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, "+", $1, $3);
    }
    |
    Expression MINUS Expression{
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, "-", $1, $3);
    }
    |
    Expression LESS Expression{
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, "<", $1, $3);
    }
    |
    Expression LESS_OR_EQUAL Expression{
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, "<=", $1, $3);
    }
    |
    Expression GREATER Expression{
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, ">", $1, $3);
    }
    |
    Expression GREATER_OR_EQUAL Expression{
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, ">=", $1, $3);
    }
    |
    Expression EQUAL Expression{
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, "=", $1, $3);
    }
    |
    Expression NOT_EQUAL Expression{
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, "<>", $1, $3);
    }
    |
    NOT Expression{
        $$ = new UnaryOperatorNode(@1.first_line, @1.first_column, "not", $2);
    }
    |
    Expression AND Expression{
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, "and", $1, $3);
    }
    |
    Expression OR Expression{
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, "or", $1, $3);
    }
    |
    IntegerAndReal{
        $$ = new ConstantValueNode(@1.first_line, @1.first_column, $1);
    }
    |
    StringAndBoolean{
        $$ = new ConstantValueNode(@1.first_line, @1.first_column, $1->val.c_str());
    }
    |
    VariableReference{
        $$ = $1;
    }
    |
    FunctionInvocation{
        $$ = $1;
    }
;

    /*
       misc
            */
Epsilon:
;

%%

void yyerror(const char *msg) {
    fprintf(stderr,
            "\n"
            "|-----------------------------------------------------------------"
            "---------\n"
            "| Error found in Line #%d: %s\n"
            "|\n"
            "| Unmatched token: %s\n"
            "|-----------------------------------------------------------------"
            "---------\n",
            line_num, current_line, yytext);
    exit(-1);
}

int main(int argc, const char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <filename> [--dump-ast]\n", argv[0]);
        exit(-1);
    }

    yyin = fopen(argv[1], "r");
    if (yyin == NULL) {
        perror("fopen() failed");
        exit(-1);
    }

    yyparse();

    if (argc >= 3 && strcmp(argv[2], "--dump-ast") == 0) {
        AstDumper ast_dumper;
        root->accept(ast_dumper);
        //root->print();
    }

    printf("\n"
           "|--------------------------------|\n"
           "|  There is no syntactic error!  |\n"
           "|--------------------------------|\n");

    delete root;
    fclose(yyin);
    yylex_destroy();
    return 0;
}