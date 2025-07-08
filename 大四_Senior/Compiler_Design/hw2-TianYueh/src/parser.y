%{
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int32_t line_num;    /* declared in scanner.l */
extern char current_line[]; /* declared in scanner.l */
extern FILE *yyin;          /* declared by lex */
extern char *yytext;        /* declared by lex */

extern int yylex(void);
static void yyerror(const char *msg);
extern int yylex_destroy(void);
%}

%token COMMA SEMICOLON COLON L_PAR R_PAR L_MID_PAR R_MID_PAR
%token PLUS MINUS MUL DIV MOD ASSIGNMENT LT LE NE GE GT EQ AND OR NOT
%token KWVAR KWARRAY KWOF KWBOOLEAN KWINTEGER KWREAL KWSTRING
%token KWTRUE KWFALSE 
%token KWDEF KWRETURN 
%token KWBEGIN KWEND
%token KWWHILE KWDO
%token KWIF KWTHEN KWELSE
%token KWFOR KWTO
%token KWPRINT KWREAD
%token ID INTEGER OCT_INTEGER FLOAT SCIENTIFIC STRING

%%

ProgramName: ID SEMICOLON
            declarations
            functions
            compound_statement
            KWEND;

declarations: declaration declarations
            | ;

functions: function functions
        |;

function: function_declaraction | function_definition;

function_header: ID L_PAR formal_arguments R_PAR COLON scalar_type
                | ID L_PAR formal_arguments R_PAR;

function_declaraction: function_header SEMICOLON;

function_definition: function_header
                    compound_statement
                    KWEND

formal_arguments: formal_argument
                | formal_arguments SEMICOLON formal_argument
                | ;

formal_argument: identifier_list COLON type;

declaration: variable_declaration
            | constant_declaration;

variable_declaration: KWVAR identifier_list COLON type SEMICOLON;

constant_declaration: KWVAR identifier_list COLON MINUS integer_literal SEMICOLON
                    | KWVAR identifier_list COLON integer_literal SEMICOLON
                    | KWVAR identifier_list COLON MINUS real_literal SEMICOLON
                    | KWVAR identifier_list COLON real_literal SEMICOLON
                    | KWVAR identifier_list COLON string_literal SEMICOLON
                    | KWVAR identifier_list COLON boolean_literal SEMICOLON;

identifier_list: ID
                | ID COMMA identifier_list;

type: scalar_type
    | array_type;

scalar_type: KWINTEGER | KWREAL | KWBOOLEAN | KWSTRING;

array_type: KWARRAY integer_literal KWOF type;

statement: simple_statement
            | conditional_statement
            | function_call_statement
            | loop_statement
            | return_statement
            | compound_statement

simple_statement: variable_reference ASSIGNMENT expression SEMICOLON
                | KWPRINT expression SEMICOLON
                | KWREAD variable_reference SEMICOLON;

conditional_statement: KWIF expression KWTHEN
                    compound_statement
                    KWELSE
                    compound_statement
                    KWEND KWIF
                    | KWIF expression KWTHEN
                    compound_statement
                    KWEND KWIF;

function_call_statement: function_call SEMICOLON;

loop_statement: KWWHILE expression KWDO
                compound_statement
                KWEND KWDO
                | KWFOR ID ASSIGNMENT integer_literal KWTO integer_literal KWDO
                compound_statement
                KWEND KWDO;

return_statement: KWRETURN expression SEMICOLON;

compound_statement: KWBEGIN
                    declarations
                    statements
                    KWEND;

statements: statements statement | ;

expression: literal_constant
            | variable_reference
            | function_call
            | binary_operation
            | unary_operation
            | L_PAR expression R_PAR;

literal_constant: integer_literal
                 | real_literal
                 | string_literal
                 | boolean_literal;

integer_literal: OCT_INTEGER | INTEGER;

real_literal: FLOAT | SCIENTIFIC;

string_literal: STRING;

boolean_literal: KWTRUE | KWFALSE;

variable_reference: ID mid_pars;

mid_pars: mid_pars L_MID_PAR expression R_MID_PAR | ;

function_call: ID L_PAR expressions R_PAR;

expressions: expressions COMMA expression | expression |;

binary_operation: expression binary_operator expression;

unary_operation: unary_operator expression;

binary_operator: PLUS | MINUS | MUL | DIV | MOD | LT | LE | NE | GE | GT | EQ | AND | OR;

unary_operator: MINUS | NOT;






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
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        exit(-1);
    }

    yyin = fopen(argv[1], "r");
    if (yyin == NULL) {
        perror("fopen() failed");
        exit(-1);
    }

    yyparse();

    fclose(yyin);
    yylex_destroy();

    printf("\n"
           "|--------------------------------|\n"
           "|  There is no syntactic error!  |\n"
           "|--------------------------------|\n");
    return 0;
}
