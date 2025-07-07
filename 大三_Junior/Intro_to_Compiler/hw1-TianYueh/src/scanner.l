%{
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define LIST_SOURCE                 appendToCurrentLine(yytext)
#define LIST_TOKEN(name)            do { LIST_SOURCE; if(opt_tok) printf("<%s>\n", #name); } while(0)
#define LIST_DELIMITER(name)        do { LIST_SOURCE; if(opt_tok) printf("<%c>\n", (name));} while(0)
#define LIST_LITERAL(name, literal) do { LIST_SOURCE; if(opt_tok) printf("<%s: %s>\n", #name, (literal)); } while(0)
#define MAX_LINE_LENG               512

uint32_t line_num = 1;
static char current_line[MAX_LINE_LENG];

static uint32_t opt_src = 1;
static uint32_t opt_tok = 1;
static char *current_line_ptr = current_line;

static void appendToCurrentLine(const char *yytext_ptr);

%}

digit [0-9]
letter [A-Za-z]
    /* Identifier should begin with letter */
identifier {letter}({digit}|{letter})*
decimalnum 0|([1-9]{digit}*)
    /* Slashes are needed for special characters */
floatnum {decimalnum}\.(0|{digit}*[1-9])
octnum 0[0-7]+
scinotation ({decimalnum}|{floatnum})([eE])([\-\+]?)({decimalnum})
    /* Anything other than \" or two \"'s */
str_const \"([^\"]|(\"\"))*\"
    /* Everything in .* would be ignored. */
pseudocomment "//&"[A-Z][\+\-].*

    /* Define a state to skip c style comments */

%x C_COMMENT
    /* Define a state to skip cpp comments, in order not to get messed up with pseudocomment */
%x CPP_COMMENT

%%

    /* Delimiters */
"," {LIST_DELIMITER(',');}
";" {LIST_DELIMITER(';');}
":" {LIST_DELIMITER(':');}
"(" {LIST_DELIMITER('(');}
")" {LIST_DELIMITER(')');}
"[" {LIST_DELIMITER('[');}
"]" {LIST_DELIMITER(']');}

    /* Operators */
"+" {LIST_TOKEN(+);}
"-" {LIST_TOKEN(-);}
"*" {LIST_TOKEN(*);}
"/" {LIST_TOKEN(/);}
"mod" {LIST_TOKEN(mod);}
":=" {LIST_TOKEN(:=);}
"<" {LIST_TOKEN(<);}
"<=" {LIST_TOKEN(<=);}
"<>" {LIST_TOKEN(<>);}
">=" {LIST_TOKEN(>=);}
">" {LIST_TOKEN(>);}
"=" {LIST_TOKEN(=);}
"and" {LIST_TOKEN(and);}
"or" {LIST_TOKEN(or);}
"not" {LIST_TOKEN(not);}

    /* Reserved Words */
"var" {LIST_TOKEN(KWvar);}
"def" {LIST_TOKEN(KWdef);}
"array" {LIST_TOKEN(KWarray);}
"of" {LIST_TOKEN(KWof);}
"boolean" {LIST_TOKEN(KWboolean);}
"integer" {LIST_TOKEN(KWinteger);}
"real" {LIST_TOKEN(KWreal);}
"string" {LIST_TOKEN(KWstring);}
"true" {LIST_TOKEN(KWtrue);}
"false" {LIST_TOKEN(KWfalse);}
"while" {LIST_TOKEN(KWwhile);}
"do" {LIST_TOKEN(KWdo);}
"if" {LIST_TOKEN(KWif);}
"then" {LIST_TOKEN(KWthen);}
"else" {LIST_TOKEN(KWelse);}
"for" {LIST_TOKEN(KWfor);}
"to" {LIST_TOKEN(KWto);}
"begin" {LIST_TOKEN(KWbegin);}
"end" {LIST_TOKEN(KWend);}
"print" {LIST_TOKEN(KWprint);}
"read" {LIST_TOKEN(KWread);}
"return" {LIST_TOKEN(KWreturn);}

    /* Implementing the declaration part */

{identifier} { LIST_LITERAL(id, yytext); } 
{decimalnum} { LIST_LITERAL(integer, yytext); }
{floatnum} { LIST_LITERAL(float, yytext); }
{octnum} { LIST_LITERAL(oct_integer, yytext); }
{scinotation} { LIST_LITERAL(scientific, yytext); }
{str_const} {
    /* MAX_LINE_LENG==512 */
    char str[MAX_LINE_LENG];
    int i=0;
    for(int x=1;x<yyleng-1;x++){
        str[i]=yytext[x];
        i++;
        if(yytext[x]=='\"'&&yytext[x+1]=='\"'){
            x++;
        }
    }
    str[i]='\0';
    LIST_LITERAL(string, str);
}

    /* Things to be discarded */

" " {LIST_SOURCE;}
\t {LIST_SOURCE;}
"/*" {BEGIN(C_COMMENT); LIST_SOURCE;}
<C_COMMENT>. {LIST_SOURCE;}
<CPP_COMMENT>. {LIST_SOURCE;}
<C_COMMENT>"*/" {BEGIN(INITIAL); LIST_SOURCE;} 
"//" {BEGIN(CPP_COMMENT); LIST_SOURCE;}

    /* Print every line as required */
<INITIAL>\n {
    if(opt_src){
        printf("%d: %s\n", line_num, current_line);
    }
    line_num++;
    current_line[0]='\0';
    current_line_ptr=current_line;
}
<C_COMMENT>\n {
    if(opt_src){
        printf("%d: %s\n", line_num, current_line);
    }
    line_num++;
    current_line[0]='\0';
    current_line_ptr=current_line;
}
<CPP_COMMENT>\n {
    BEGIN(INITIAL);
    if(opt_src){
        printf("%d: %s\n", line_num, current_line);
    }
    line_num++;
    current_line[0]='\0';
    current_line_ptr=current_line;
}

    /* Dealing with the Pseudo Comments OMG */
{pseudocomment} {
    if(yytext[3]=='S'){
        if(yytext[4]=='+'){
            opt_src=1;
        }
        else if(yytext[4]=='-'){
            opt_src=0;
        }
    }
    else if(yytext[3]=='T'){
        if(yytext[4]=='+'){
            opt_tok=1;
        }
        else if (yytext[4]=='-'){
            opt_tok=0;
        }
    }
    LIST_SOURCE;
}







    /* Catch the character which is not accepted by rules above */
. {
    printf("Error at line %d: bad character \"%s\"\n", line_num, yytext);
    exit(-1);
}

%%

static void appendToCurrentLine(const char *yytext_ptr) {
    while (*yytext_ptr) {
        *current_line_ptr = *yytext_ptr;
        ++current_line_ptr;
        ++yytext_ptr;
    }
    *current_line_ptr = '\0';
}

int main(int argc, const char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s [filename]\n", argv[0]);
        exit(-1);
    }

    yyin = fopen(argv[1], "r");
    if (yyin == NULL) {
        fprintf(stderr, "Open file error\n");
        exit(-1);
    }

    yylex();

    fclose(yyin);
    /* Free resources used by the scanner; otherwise, unused data structures
    will persist throughout the successive compiler phases. */
    yylex_destroy();

    return 0;
}
