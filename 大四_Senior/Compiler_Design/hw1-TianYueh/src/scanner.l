%option never-interactive
%option nounput
%option noinput

%{
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_LINE_LEN 512
/* Code runs each time a token is matched. */
#define YY_USER_ACTION updateCurrentLine(yytext);

uint32_t line_num = 1;
uint32_t col_num = 1;
static uint32_t opt_src = 1;
static uint32_t opt_tok = 1;
static char current_line[MAX_LINE_LEN];

static void updateCurrentLine(const char *source);
static void listToken(const char *name);
static void listLiteral(const char *name, const char *literal);

%}

digit [0-9]
letter [a-zA-Z]
identifier {letter}({letter}|{digit})*
decinum 0|([1-9]{digit}*)
floatnum {decinum}\.(0|{digit}*[1-9])
octnum 0[0-7]+
scinotation (([1-9]{digit}*)|([1-9]{digit}*)\.(0|{digit}*[1-9]))([eE])([\+\-]?)({decinum})
str_const \"([^\"]|\"\")*\"
    /*Ignore things in .*/
pseudocomment "//&"[A-Z][\+\-].*

    /*Skip the comment*/

%x C_COMMENT
%x CPP_COMMENT

%%

    /* Delimiters */

"(" { listToken("("); }
")" { listToken(")"); }
"[" { listToken("["); }
"]" { listToken("]"); }
"," { listToken(","); }
";" { listToken(";"); }
":" { listToken(":"); }

    /* Operators */

"+" { listToken("+"); }
"-" { listToken("-"); }
"*" { listToken("*"); }
"/" { listToken("/"); }
"mod" { listToken("mod"); }
":=" { listToken(":="); }
"<" { listToken("<"); }
">" { listToken(">"); }
"<=" { listToken("<="); }
"<>" { listToken("<>"); }
">=" { listToken(">="); }
"=" { listToken("="); }
"and" { listToken("and"); }
"or" { listToken("or"); }
"not" { listToken("not"); }

    /* Reserved words */

"var" { listToken("KWvar"); }
"def" { listToken("KWdef"); }
"array" { listToken("KWarray"); }
"of" { listToken("KWof"); }
"boolean" { listToken("KWboolean"); }
"integer" { listToken("KWinteger"); }
"real" { listToken("KWreal"); }
"string" { listToken("KWstring"); }
"true" { listToken("KWtrue"); }
"false" { listToken("KWfalse"); }
"while" { listToken("KWwhile"); }
"do" { listToken("KWdo"); }
"if" { listToken("KWif"); }
"then" { listToken("KWthen"); }
"else" { listToken("KWelse"); }
"for" { listToken("KWfor"); }
"to" { listToken("KWto"); }
"begin" { listToken("KWbegin"); }
"end" { listToken("KWend"); }
"print" { listToken("KWprint"); }
"read" { listToken("KWread"); }
"return" { listToken("KWreturn"); }

    /* Literals */

{identifier} { listLiteral("id", yytext); }
{decinum} { listLiteral("integer", yytext); }
{floatnum} { listLiteral("float", yytext); }
{octnum} { listLiteral("oct_integer", yytext); }
{scinotation} { listLiteral("scientific", yytext); }
{str_const} {
    char str[MAX_LINE_LEN];
    int i = 0;
    for(int x = 1;x<yyleng-1;x++){
        str[i] = yytext[x];
        i++;
        if(yytext[x] == '\"' && yytext[x+1] == '\"'){
            x++;
        }
    }
    str[i] = '\0';
    listLiteral("string", str);
}


    /* Something to discard or to change state */

" " {}
\t {}
"/*" {BEGIN C_COMMENT;}
<C_COMMENT>. {}
<CPP_COMMENT>. {}
<C_COMMENT>"*/" {BEGIN INITIAL;}
<CPP_COMMENT>\n {BEGIN INITIAL;}
"//" {BEGIN CPP_COMMENT;}

<INITIAL,C_COMMENT>\n {}

    /* Pseudocomment */

{pseudocomment} {
    if(yytext[3] == 'S'){
        if(yytext[4] == '+'){
            opt_src = 1;
        }
        else if(yytext[4] == '-'){
            opt_src = 0;
        }
    }
    else if(yytext[3] == 'T'){
        if(yytext[4] == '+'){
            opt_tok = 1;
        }
        else if(yytext[4] == '-'){
            opt_tok = 0;
        }
    }
}




    /* Catch the character which is not accepted by rules above */
. {
    printf("Error at line %d: bad character \"%s\"\n", line_num, yytext);
    exit(-1);
}

%%

/** @note The line is printed out and flushed when a newline character is encountered. */
static void updateCurrentLine(const char *source) {
    /* col_num is one-based */
    for (const char *c = source; *c; ++c) {
        if (*c == '\n') {
            if (opt_src) {
                printf("%d: %s\n", line_num, current_line);
                
            }
            ++line_num;
            col_num = 1;
            current_line[0] = '\0';
        }
        else {
            if (col_num < MAX_LINE_LEN) {
                current_line[col_num - 1] = *c;
                current_line[col_num] = '\0';
                //printf("DEBUG: %s\n DEBUG: \"%c\"\n", current_line, *c);
                //printf("DEBUG: %d\n", col_num);
            } else {
                /* Truncate silently; doesn't affect the program's correctness. */
            }
            ++col_num;
        }
    }
}

static void listToken(const char *name) {
    if (opt_tok) {
        printf("<%s>\n", name);
    }
}

static void listLiteral(const char *name, const char *literal) {
    if (opt_tok) {
        printf("<%s: %s>\n", name, literal);
    }
}

/** @note This function is not required if the input file is guaranteed to end
 * with a newline. However, students may find it useful to handle the case where
 * the input file does not end with a newline, as it has been reported several
 * times in the past.
 */
int yywrap(void) {
    /* If the file is not ended with a newline, fake it to print out the last line. */
    if (col_num > 1) {
        updateCurrentLine("\n");
    }
    /* no more input file */
    return 1;
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
