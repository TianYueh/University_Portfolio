# hw4 report

|||
|-:|:-|
|Name|房天越|
|ID|110550085|

## How much time did you spend on this project

24 hours, maybe more.

## Project overview

In this project, we implement semantic analysis based on the AST tree we constructed in HW3. To do so, first we construct symbol table, symbol entry, and symbol manager. These are implemented in Semantic Analyzer files in my implementation, which is used to traverse the children. 
A symbol entry is an entry that consists of the data of a symbol, including its name, kind, level, type, and attribute. A symbol table consists a lot of symbol entries, all the entries that are in the same scope will be put under symbol_table which is define in the class SymbolTable. And, a symbol manager consists of many tables, all the scopes would be put on the stack according to their levels, it is also capable of recoding the loop_vars and the constants.
The parser and the scanner also need modification, in the parser, I modified the last output part to let it not show the non-error message when there are errors, and in the scanner, I made it able to record the source codes for error output and made a new option "D" to make it able to control whether to dump the table or not.
Then, I made it able to traverse along the AST tree, dump tables, check whether there are errors, and output error messages while traversing. The error messages are defined according to the error-message.md. Since this is the most complicated part, and the behaviors are mostly written in the spec, I would not explain too many details here.

## What is the hardest you think in this project

This project is even more complicated than HW3 in my opinion, even realizing what this project requires us to do took me a lot of time, and the need to maintain a lot of details is also quite difficult and requires concentration.
I think the hardest part of this project is to build the symbol table, because there are a lot of things to concern when considering scope and their interactions, just passing the first testcase took me a lot of time. Also, it is also difficult to implement error checking because there are a lot of details to notice, and a lot of flags are required while implementing, or there might be error output when there should not be, or there would not be error output when there should be. Lastly, I also want to mention that the conversion between string and const char* is also an important part to notice. At first I tried to compare them directly, and I got it wrong but did not know why, it also took me some time to figure that out. 

## Feedback to T.A.s

As I said above, this project is way more complicated than HW3, probably the most difficult in this semester. So I would like to say thank you for spending your time maintaining such a huge and complicated project, and marking our homework. But I also want to say that as the final exams approach, I have a hard time dealing with it, and find that other subjects are getting messed up. Hope that I could pass all the subjects successfully QQ. Wish you a Merry Christmas and a Happy New Year.
