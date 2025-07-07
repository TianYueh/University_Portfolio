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

%code requires {
    #include <vector>
    class AstNode;
    class DeclNode;
    class CompoundStatementNode;
    struct Unary_Op;
    struct Binary_Op;
    /*struct Constant_Value*/
    struct Name;
}

    /* For yylval */
%union {
    /* basic semantic value */
    char *identifier;
    char *const_str;
    int const_int;
    float const_real;
    //CompoundStatementNode *compound_stmt_ptr;

    AstNode *node;

    std::vector<AstNode*> *node_list;
    std::vector<Name*> *name_list;
    struct Constant_Value *const_val;

};

%type <identifier> ProgramName ID FunctionName
%type <node> CompoundStatement
%type <node_list> DeclarationList Declarations FunctionList Functions FormalArgList FormalArgs StatementList
%type <node_list> ExpressionList Expressions ArrRefList ArrRefs Statements
%type <node> Declaration Function FunctionDeclaration FunctionDefinition FormalArg Statement
%type <node> Simple Condition While For Return FunctionCall FunctionInvocation Expression VariableReference ElseOrNot
%type <name_list> IdList


%type <const_str> ReturnType Type ScalarType ArrType ArrDecl
%type <const_str> VAR STRING_LITERAL TRUE FALSE INTEGER REAL STRING BOOLEAN ARRAY

%type <const_val> LiteralConstant StringAndBoolean IntegerAndReal
%type <const_int> NegOrNot INT_LITERAL
%type <const_real> REAL_LITERAL

    /* Follow the order in scanner.l */

    /* Delimiter */
%token COMMA SEMICOLON COLON
%token L_PARENTHESIS R_PARENTHESIS
%token L_BRACKET R_BRACKET

    /* Operator */
%token ASSIGN
%left OR
%left AND
%right NOT
%left LESS LESS_OR_EQUAL EQUAL GREATER GREATER_OR_EQUAL NOT_EQUAL
%left PLUS MINUS
%left MULTIPLY DIVIDE MOD
%right UNARY_MINUS

    /* Keyword */
%token ARRAY BOOLEAN INTEGER REAL STRING
%token END BEGIN_ /* Use BEGIN_ since BEGIN is a keyword in lex */
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


Program:
    ProgramName SEMICOLON
    /* ProgramBody */
    DeclarationList FunctionList CompoundStatement
    /* End of ProgramBody */
    END {
        root = new ProgramNode(@1.first_line, @1.first_column,
                               $1, "void", $3, $4, $5);
        free($1);
    }
;

ProgramName:
    ID
;

DeclarationList:
    Epsilon
    {
        $$ = new std::vector<AstNode*>();
    }
    |
    Declarations
    {
        $$ = $1;
    }
;

Declarations:
    Declaration
    {
        std::vector<AstNode*> *decls = new std::vector<AstNode*>;
        if($1 != nullptr){
            decls->push_back($1);
        }
        $$ = decls;
    }
    |
    Declarations Declaration
    {
        $$ = $1;
        if($2 != nullptr){
            $1->push_back($2);
        }
        
        
    }
;

FunctionList:
    Epsilon
    {
        $$ = new std::vector<AstNode*>();
    }
    |
    Functions
    {
        $$ = $1;
    }
;

Functions:
    Function
    {
        std::vector<AstNode*> *func_list = new std::vector<AstNode*>;
        func_list->push_back($1);
        $$ = func_list;
    }
    |
    Functions Function
    {
        $$ = $1;
        $$->push_back($2);
    }
;

Function:
    FunctionDeclaration
    {
        $$ = $1;
    }
    |
    FunctionDefinition
    {
        $$ = $1;
    }
;

FunctionDeclaration:
    FunctionName L_PARENTHESIS FormalArgList R_PARENTHESIS ReturnType SEMICOLON
    {
        $$ = new FunctionNode(@1.first_line, @1.first_column, $1, $5, $3, nullptr);
    }
;

FunctionDefinition:
    FunctionName L_PARENTHESIS FormalArgList R_PARENTHESIS ReturnType
    CompoundStatement
    END
    {
        $$ = new FunctionNode(@1.first_line, @1.first_column, $1, $5, $3, $6);
    }
;

FunctionName:
    ID
;

FormalArgList:
    Epsilon
    {
        $$ = new std::vector<AstNode*>();
    }
    |
    FormalArgs{
        $$ = $1;
    }
;

FormalArgs:
    FormalArg
    {
        std::vector<AstNode*> *decls = new std::vector<AstNode*>;
        decls->push_back($1);
        $$ = decls;
    }
    |
    FormalArgs SEMICOLON FormalArg
    {
        $$ = $1;
        $$->push_back($3);
    }
;

FormalArg:
    IdList COLON Type
    {
        std::vector<AstNode*> *var_list = new std::vector<AstNode*>;
        for(uint32_t i = 0;i<$1->size();i++){
            VariableNode *va = new VariableNode($1->at(i)->line, $1->at(i)->col, $1->at(i)->id, $3, nullptr);
            var_list->push_back(va);
        }
        $$ = new DeclNode(@1.first_line, @1.first_column, var_list);
    }
;

IdList:
    ID
    {
        $$ = new std::vector<Name*>;
        Name *n = new Name;
        n->id = $1;
        n->line = @1.first_line;
        n->col = @1.first_column;
        $$->push_back(n);
        
    }
    |
    IdList COMMA ID
    {
        
        $$ = $1;
        Name *n = new Name;
        n->id = $3;
        n->line = @3.first_line;
        n->col = @3.first_column;
        $$->push_back(n);
        
    }
;

ReturnType:
    COLON ScalarType
    {
        $$ = $2;
    }
    |
    Epsilon
    {
        char s[100];
        strcpy(s, "void");
        $$ = s;
    }
;

    /*
       Data Types and Declarations
                                   */

Declaration:
    VAR IdList COLON Type SEMICOLON
    {
        std::vector<AstNode*> *var_list = new std::vector<AstNode*>;
        for(uint32_t i=0;i<$2->size();i++){
            VariableNode *va = new VariableNode($2->at(i)->line, $2->at(i)->col, $2->at(i)->id, $4, nullptr);
            var_list->push_back(va);
        }
        $$ = new DeclNode(@1.first_line, @1.first_column, var_list);
        
    }
    |
    VAR IdList COLON LiteralConstant SEMICOLON
    {
        std::vector<AstNode*> *var_list = new std::vector<AstNode*>;
        char s[100];
        for(uint32_t i=0;i<$2->size();i++){
            ConstantValueNode *c = new ConstantValueNode($4->line, $4->col, *($4));
            if($4->int_type){
                strcpy(s, "integer");
            }
            else if($4->real_type){
                strcpy(s, "real");
            }
            else if($4->str_type){
                strcpy(s, "string");
            }
            else if($4->bool_type){
                strcpy(s, "boolean");
            }
            VariableNode *va = new VariableNode($2->at(i)->line, $2->at(i)->col, $2->at(i)->id, s, c);
            var_list->push_back(va);
        }
        $$ = new DeclNode(@1.first_line, @1.first_column, var_list);
        
    }
;

Type:
    ScalarType
    {
        $$ = $1;
    }
    |
    ArrType
    {
        $$ = $1;
    }
;

ScalarType:
    INTEGER
    {
        char s[100];
        strcpy(s, "integer");
        $$ = s;
    }
    |
    REAL
    {
        char s[100];
        strcpy(s, "real");
        $$ = s;
    }
    |
    STRING
    {
        char s[100];
        strcpy(s, "string");
        $$ = s;
    }
    |
    BOOLEAN
    {
        char s[100];
        strcpy(s, "boolean");
        $$ = s;
    }
;

ArrType:
    ArrDecl ScalarType
    {
        char s1[100];
        char s2[100];
        char s3[100];
        strcpy(s1, $2);
        strcpy(s3, $1);
        strcpy(s2," ");
        strcat(s1, s2);
        strcat(s1, s3);
        $$ = s1;
        
    }
;

ArrDecl:
    ARRAY INT_LITERAL OF
    {
        
        char s1[100];
        char s2[100];
        strcpy(s1, "[");
        int val = $2;
        sprintf(s2, "%d", val);
        strcat(s1, s2);
        strcpy(s2, "]");
        strcat(s1, s2);
        $$ = s1;
        
    }
    |
    ArrDecl ARRAY INT_LITERAL OF
    {
        
        char s1[100];
        char s2[100];
        strcpy(s1, $1);
        strcpy(s2, "[");
        strcat(s1, s2);
        int val = $3;
        sprintf(s2, "%d", val);
        strcat(s1, s2);
        strcpy(s2, "]");
        strcat(s1, s2);
        $$ = s1;
        
    }
;

LiteralConstant:
    NegOrNot INT_LITERAL
    {
        
        $$ = new Constant_Value;
        $$ ->int_value = $1 * $2;
        $$->int_type = true;
        if($1 == 1){
            $$->line = @2.first_line;
            $$->col = @2.first_column;
        }
        else{
            $$->line = @1.first_line;
            $$->col = @1.first_column;
        }
        
    }
    |
    NegOrNot REAL_LITERAL
    {
        
        $$ = new Constant_Value;
        $$ ->real_value = $1 * $2;
        $$->real_type = true;
        if($1 == 1){
            $$->line = @2.first_line;
            $$->col = @2.first_column;
        }
        else{
            $$->line = @1.first_line;
            $$->col = @1.first_column;
        }
        
    }
    |
    StringAndBoolean
    {
        
        $$ = $1;
        $$->line = @1.first_line;
        $$->col = @1.first_column;
        
    }
;

NegOrNot:
    Epsilon
    {
        
        $$ = 1;
        
    }
    |
    MINUS %prec UNARY_MINUS
    {
        
        $$ = -1;
        
    }
;

StringAndBoolean:
    TRUE
    {
        
        $$ = new Constant_Value;
        $$->str_value = "true";
        $$->bool_type = true;
        $$->str_type = false;
    }
    |
    FALSE
    {
        
        $$ = new Constant_Value;
        $$->str_value = "false";
        $$->bool_type = true;
        $$->str_type = false;
        
    }
    |
    STRING_LITERAL
    {
        $$ = new Constant_Value;
        $$->str_value = $1;
        $$->str_type = true;

        
    }
;

IntegerAndReal:
    INT_LITERAL
    {
        $$ = new Constant_Value;
        $$->int_value = $1;
        $$->int_type = true;
    }
    |
    REAL_LITERAL
    {
        $$ = new Constant_Value;
        $$->real_value = $1;
        $$->real_type = true;
    }
    
;

    /*
       Statements
                  */

Statement:
    CompoundStatement
    {
        $$ = $1;
    }
    |
    Simple
    {
        $$ = $1;
    }
    |
    Condition
    {
        $$ = $1;
    }
    |
    While
    {
        $$ = $1;
    }
    |
    For
    {
        $$ = $1;
    }
    |
    Return
    {
        $$ = $1;
    }
    |
    FunctionCall
    {
        $$ = $1;
    }
;

CompoundStatement:
    BEGIN_
    DeclarationList
    StatementList
    END {
        $$ = new CompoundStatementNode(@1.first_line, @1.first_column, $2, $3);
    }
;

Simple:
    VariableReference ASSIGN Expression SEMICOLON
    {
        $$ = new AssignmentNode(@2.first_line, @2.first_column, $1, $3);
    }
    |
    PRINT Expression SEMICOLON
    {
        $$ = new PrintNode(@1.first_line, @1.first_column, $2);
    }
    |
    READ VariableReference SEMICOLON
    {
        $$ = new ReadNode(@1.first_line, @1.first_column, $2);
    }
;

VariableReference:
    ID ArrRefList
    {
        //printf("FOUND VARREF\n");
        $$ = new VariableReferenceNode(@1.first_line, @1.first_column, $1, $2);
    }
;

ArrRefList:
    Epsilon
    {
        $$ = new std::vector<AstNode*>();
    }
    |
    ArrRefs
    {
        $$ = $1;
    }
;

ArrRefs:
    L_BRACKET Expression R_BRACKET
    {
        std::vector<AstNode*>* v = new std::vector<AstNode*>;
        v->push_back($2);
        $$ = v;
    }
    |
    ArrRefs L_BRACKET Expression R_BRACKET
    {
        $$ = $1;
        $$->push_back($3);
    }
;

Condition:
    IF Expression THEN
    CompoundStatement
    ElseOrNot
    END IF
    {
        //CompoundStatement is NULL
        {
            if($4 == NULL){
                $$ = new IfNode(@1.first_line, @1.first_column, $2, $4, nullptr);
            }
            else{
                $$ = new IfNode(@1.first_line, @1.first_column, $2, $4, $5);
            }
        }
    }
;

ElseOrNot:
    ELSE
    CompoundStatement
    {
        $$ = $2;
    }
    |
    Epsilon
    {
        $$ = nullptr;
    }
;

While:
    WHILE Expression DO
    CompoundStatement
    END DO
    {
        $$ = new WhileNode(@1.first_line, @1.first_column, $2, $4);
    }
;

For:
    FOR ID ASSIGN INT_LITERAL TO INT_LITERAL DO
    CompoundStatement
    END DO
    {
        std::vector<AstNode*>* var_list = new std::vector<AstNode*>;
        VariableNode *va = new VariableNode(@2.first_line, @2.first_column, $2, "integer", nullptr);
        var_list->push_back(va);
        DeclNode* d = new DeclNode(@2.first_line, @2.first_column, var_list);

        VariableReferenceNode* vrn = new VariableReferenceNode(@2.first_line, @2.first_column, $2, nullptr);
        Constant_Value *cv = new Constant_Value;
        cv->int_value = $4;
        cv->int_type = true;
        ConstantValueNode* c = new ConstantValueNode(@4.first_line, @4.first_column, *(cv));
        AssignmentNode* a = new AssignmentNode(@3.first_line, @3.first_column, vrn, c);
    
        Constant_Value *cv2 = new Constant_Value;
        cv2->int_value = $6;
        cv2->int_type = true;
        ConstantValueNode *c2 = new ConstantValueNode(@6.first_line, @6.first_column, *(cv2));
    
        $$ = new ForNode(@1.first_line, @1.first_column, d, a, c2, $8);
    }
;

Return:
    RETURN Expression SEMICOLON
    {
        $$ = new ReturnNode(@1.first_line, @1.first_column, $2);
    }
;

FunctionCall:
    FunctionInvocation SEMICOLON
    {
        $$ = $1;
    }
;

FunctionInvocation:
    ID L_PARENTHESIS ExpressionList R_PARENTHESIS
    {
        $$ = new FunctionInvocationNode(@1.first_line, @1.first_column, $1, $3);
    }
;

ExpressionList:
    Epsilon
    {
        $$ = new std::vector<AstNode*>();
    }
    |
    Expressions
    {
        $$ = $1;
    }
;

Expressions:
    Expression
    {
        
        std::vector<AstNode*>* v = new std::vector<AstNode*>;
        v->push_back($1);
        $$ = v;
    }
    |
    Expressions COMMA Expression
    {
    $$ = $1;
    $$->push_back($3);
    }
;

StatementList:
    Epsilon
    {
        $$ = new std::vector<AstNode*>();
    }
    |
    Statements
    {
        $$ = $1;
    }
;

Statements:
    Statement
    {
        std::vector<AstNode*>* v = new std::vector<AstNode*>;
        if($1 != nullptr){
            v->push_back($1);
        }
        $$ = v;
    }
    |
    Statements Statement
    {
        $$ = $1;
        if($2 != nullptr){
            $$->push_back($2);
        }
    }
;

Expression:
    L_PARENTHESIS Expression R_PARENTHESIS
    {
        $$ = $2;
    }
    |
    MINUS Expression %prec UNARY_MINUS
    {
        //printf(" ffff ffffffF\n");
        Unary_Op u;
        u.neg = true;
        $$ = new UnaryOperatorNode(@1.first_line, @1.first_column, u, $2);
    }
    |
    Expression MULTIPLY Expression
    {
        Binary_Op b;
        b.mul = true;
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, b, $1, $3);
    }
    |
    Expression DIVIDE Expression
    {
        Binary_Op b;
        b.div = true;
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, b, $1, $3);
    }
    |
    Expression MOD Expression
    {
        Binary_Op b;
        b.mod = true;
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, b, $1, $3);
    }
    |
    Expression PLUS Expression
    {
        Binary_Op b;
        b.add = true;
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, b, $1, $3);
    }
    |
    Expression MINUS Expression
    {
        Binary_Op b;
        b.sub = true;
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, b, $1, $3);
    }
    |
    Expression LESS Expression
    {
        Binary_Op b;
        b.lt = true;
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, b, $1, $3);
    }
    |
    Expression LESS_OR_EQUAL Expression
    {
        Binary_Op b;
        b.le = true;
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, b, $1, $3);
    }
    |
    Expression GREATER Expression
    {
        Binary_Op b;
        b.gt = true;
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, b, $1, $3);
    }
    |
    Expression GREATER_OR_EQUAL Expression
    {
        Binary_Op b;
        b.ge = true;
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, b, $1, $3);
    }
    |
    Expression EQUAL Expression
    {
        Binary_Op b;
        b.eq = true;
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, b, $1, $3);
    }
    |
    Expression NOT_EQUAL Expression
    {
        Binary_Op b;
        b.ne = true;
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, b, $1, $3);
    }
    |
    NOT Expression
    {
        Unary_Op u;
        u.NOT = true;
        $$ = new UnaryOperatorNode(@1.first_line, @1.first_column, u, $2);
    }
    |
    Expression AND Expression
    {
        Binary_Op b;
        b.AND = true;
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, b, $1, $3);
    }
    |
    Expression OR Expression
    {
        Binary_Op b;
        b.OR = true;
        $$ = new BinaryOperatorNode(@2.first_line, @2.first_column, b, $1, $3);
    }
    |
    IntegerAndReal
    {
        //printf("FOUND INTANDREAL\n");
        $$ = new ConstantValueNode(@1.first_line, @1.first_column, *($1));
    }
    |
    StringAndBoolean
    {
        $$ = new ConstantValueNode(@1.first_line, @1.first_column, *($1));
    }
    |
    VariableReference
    {
        $$ = $1;
    }
    |
    FunctionInvocation
    {
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
