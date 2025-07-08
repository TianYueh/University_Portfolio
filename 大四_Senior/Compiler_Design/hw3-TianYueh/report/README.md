# hw3 report

|      |                 |
| ---: | :-------------- |
| Name | 房天越           |
|   ID | 110550085       |

## How much time did you spend on this project

More than 15 hours.

## Project overview

In this project, we need to construct AST grammars for the parser, and modify the scanner and AST files, including the hpp and cpp files to implement the traversal of the AST trees. 

In the scanner, we use strtol, atof, and strndup functions to pass the values to yylval.

In the parser, we implement the AST grammar for the trees.
 1. First, we modify the %token part. We use %left and %right to specify the precedence of the operators.
 2. We modify the %code require part to specify the libraries and the nodes that would be used in the parser. Also, we construct a struct in this part, which allows us to declare and store types more easily.
 3. In the %code union part, we define the types for the non-terminals and yylval can use.
 4. In %type part, assign types to the non-terminals.
 5. Finally, we construct the grammar rules for each grammar defined in the parser, so that we can create new nodes and pass the required values to the codes.

Next, we move on to discuss the grammars in the include and lib files.
 1. For each hpp files, because we want to use the Visitor Pattern, we need to include the "visitor/AstNodeVisitor.hpp" for them.
 2. According to what nodes are used in each of the code, we also need to include the corresponding hpp files for them.
 3. We also need to extend to declaration of each file to satisfy the requirements of each node.
 4. If variables are needed, we also need to declare them.
 5. We then add accept and visitChildNodes functions for each of them.
 6. For the AstDumper file, we need to specify the functions needed to get the values for the nodes.

## What is the hardest you think in this project

While I actually have taken the course once, I still had a hard time finishing this homework because it is very difficult to debug.
The number of files to maintain is overwhelming. Sometimes some little mistakes can lead to UAF. 
I tried to use const char* for many part that are now using std::string, however, UAF generates. I had no idea why and asked ChatGPT. It says that using const char* and using .c_str() might cause dangling pointer problem. I fixed that by using strdup(). I learned a lesson from this.
Also in the function code, it is difficult to implement the rule during the declaration.


## Feedback to T.A.s

With this homework, I reviewed the concept of OOP once again. Also, it is an important experience to maintain such a big project and construct the rules step by step. Thank you for creating this homework and thank you for reading this with my poor English :D.
