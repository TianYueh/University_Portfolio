# hw1 report

|Field|Value|
|-:|:-|
|Name|房天越|
|ID|110550085|

## How much time did you spend on this project

About 6 hours.

## Project overview

Describe the project structure and how you implemented it.
This project mainly includes two parts. 
The first part is to define the regular expressions of the elements, such as digit, letter, and something more complicated. The second part is to define the rules for the compiler to do when encountering different elements, it may need to print it out, print it with something added such as KW, discard it or change the states.
In the first part, I write the regular expressions of the elements in the order of the GitHub repo writes.
And in the second part, I write the actions for the compiler to do, including three kinds of list, yytext, and also control the opt's when encountering pseudocomments.

## What is the hardest you think in this project

The hardest part I think is to deal with the conflict between the real comments and the pseudocomments. At first I tried to use two regular expressions to deal with the cpp style comments and the pseudocomments, but I ended up finding that all the pseudocomments would be viewed as cpp comments, which leads to errors. In the end, I decided to use a state to deal with the cpp comments like the way I deal with c style comments.
Another difficult part I also want to mention is think about how to deal with the strings, I tried a lot of times to make it correct.

## Feedback to T.A.s

> Please help us improve our assignment, thanks.
