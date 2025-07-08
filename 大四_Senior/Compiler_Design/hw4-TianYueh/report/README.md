# hw4 report

|||
|-:|:-|
|Name|房天越|
|ID|110550085|

## How much time did you spend on this project

More than 20 hours.

## Project overview

In this project, we aim to implement Semantic Analysis based on the AST trees we constructed in HW3. 

First, we add an option "D" for dumping in the scanner.l, and use int32_t opt_dump to record it.  
Second, we modify the parser.y to let it not print the Non-Error message when there is indeed errors.  

Let's move on to discuss the Semantic Analysis part.  

To do so, we implement Symbol Table, Symbol Entry, and Symbol Manager (and some other Managers).  
A Symbol Entry is an entry consisting the data of a symbol, including its name, kind, level, type, and attribute.  
A Symbol Table consists many symbol entries, all the entries in the same scope will be put under a vector defined in the class SymbolTable. Also, it provides insertion, lookup, and printing. 
The Symbol Manager maintains the scope stack, enabling cross-scope lookups and error detection.  
The Context Manager stores expression types and function call state during AST traversal.   
The Return Type Manager formats function return types for return validation.  
Some methods are defined in the Semantic Analyzer class.  
  
For the Symbol and Scope Management, we generally follow the working flow of the TODO given by the TAs.  
1. Push a new symbol table if this node forms a scope by calling symbolManager.pushScope().  
2. Insert the symbol into current symbol table if this node is related to declaration by calling SymbolTable::addSymbol().  
3. Traverse child nodes of this node with visitChildNodes(*this).  
4. Perform semantic analyses of this node.  
5. Pop the symbol table pushed at the 1st step with symbolManager.popScope().  
For the semantic analaysis part, we need to notice that there might be errors, we need to adjust some flags to indicate the current status and ask the Error Printer to print the error messages defined in the Error files.  

## What is the hardest you think in this project

This project is even more complicated than HW3. A lot of details are required to be maintained, this is the hardest part I think.  
I actually wrote this once, there was no error printer then, the error messages needs to be written by ourselves, that was even more difficult.  
Despite the fact that there is no need to write the error messages by ourselves, it is still difficult to concentrate and maintain the overwhelming details in the files, including the symbol table and some semantic analysis parts. Even a miss in a small flag would result in failure.  

## Feedback to T.A.s

Thank you again for creating and scoring the homeworks, they're definitely not easy tasks.  
I would like to say that using the Error files and Error Printer is a nice inspiration! Compared with my previous time implementing this, it really saves a lot of time and make the whole project more readable and explicit. Thanks again!  
