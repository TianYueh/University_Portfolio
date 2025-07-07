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


%token COMMA SEMICOLON COLON LEFT_PAR RIGHT_PAR LEFT_JUNGGUAHAO RIGHT_JUNGGUAHAO
%token ADD MINUS MULTIPLY DIVIDE MODULO ASSIGNMENT LT LE NE GE GT EQ AND OR NOT
%token KWVAR KWARRAY KWOF KWBOOLEAN KWINTEGER KWREAL KWSTRING
%token KWTRUE KWFALSE
%token KWDEF KWRETURN
%token KWBEGIN KWEND
%token KWWHILE KWDO
%token KWIF KWTHEN KWELSE
%token KWFOR KWTO
%token KWPRINT KWREAD
%token ID INTEGER OCT_INTEGER FLOAT SCIENTIFIC STRING

%left ADD MINUS MULTIPLY DIVIDE MODULO LT LE NE GE GT EQ AND OR NOT

%%

ProgramName: ID SEMICOLON
            var_const_decl_list
            func_decl_def_list
            compound
            KWEND
            ;

function_decl: ID LEFT_PAR formal_arg_list RIGHT_PAR COLON scalar_type SEMICOLON
            | ID LEFT_PAR formal_arg_list RIGHT_PAR COLON SEMICOLON
            ;

function_def: ID LEFT_PAR formal_arg_list RIGHT_PAR COLON scalar_type
            compound
            KWEND
            | ID LEFT_PAR formal_arg_list RIGHT_PAR
            compound
            KWEND
            ;

func_decl_def_list: func_decl_def_list function_def | func_decl_def_list function_decl | ;

formal_arg: id_list COLON type ;

formal_arg_list: formal_arg_list formal_arg | ;

id_list: ID id_list_ ;

id_list_: id_list_ COMMA ID | ;

var_decl: KWVAR id_list COLON scalar_type SEMICOLON 
            | KWVAR id_list COLON aio KWARRAY int_const KWOF type SEMICOLON;

aio: aio KWARRAY int_const KWOF | ;

int_const: INTEGER ;

const_decl: KWVAR id_list COLON literal_const SEMICOLON ;

compound: KWBEGIN
            var_const_decl_list
            statement_list
            KWEND
        ;

statement: compound
            | simple
            | conditional
            | while_statement
            | for_statement
            | return_statement
            | function_call
            ;

statement_list: statement_list statement | ;

simple: var_reference ASSIGNMENT expression SEMICOLON 
        | KWPRINT expression SEMICOLON
        | KWREAD var_reference SEMICOLON
        ;

var_reference: ID
                | array_ref 
                ;

array_ref: ID expression_list ;

expression_list: expression_list LEFT_JUNGGUAHAO expression RIGHT_JUNGGUAHAO | ;

conditional: KWIF expression KWTHEN
            compound
            KWELSE
            compound
            KWEND KWIF
            | KWIF expression KWTHEN
            compound
            KWEND KWIF
            ;

while_statement: KWWHILE expression KWDO
                    compound
                    KWEND KWDO
                    ;

for_statement: KWFOR ID ASSIGNMENT int_const KWTO int_const KWDO
                compound
                KWEND KWDO

return_statement: KWRETURN expression SEMICOLON;

function_call: ID LEFT_PAR expression_list_by_commas RIGHT_PAR SEMICOLON ;
function_call_without_semicolon: ID LEFT_PAR expression_list_by_commas RIGHT_PAR ;

expression_list_by_commas: expression expression_list_by_commas_ | ;

expression_list_by_commas_: COMMA expression expression_list_by_commas_ | ;

expression: literal_const
            | var_reference
            | function_call_without_semicolon
            | MINUS expression %prec MULTIPLY
            | LEFT_PAR expression RIGHT_PAR
            | expression MULTIPLY expression
            | expression DIVIDE expression
            | expression MODULO expression
            | expression ADD expression
            | expression MINUS expression
            | expression LT expression
            | expression LE expression
            | expression NE expression
            | expression GE expression
            | expression GT expression
            | expression EQ expression
            | expression AND expression
            | expression OR expression 
            | NOT expression
            ;

literal_const: INTEGER | OCT_INTEGER | FLOAT | SCIENTIFIC | STRING | KWTRUE | KWFALSE ;

scalar_type: KWINTEGER | KWREAL | KWSTRING | KWBOOLEAN ;

var_const_decl_list: var_const_decl_list const_decl
                    | var_const_decl_list var_decl
                    |
                    ;

type: scalar_type | array_ref ;

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
