# hw3 report

|||
|-:|:-|
|Name|房天越|
|ID|110550085|

## How much time did you spend on this project

Like 20 hours, maybe more.

## Project overview

There are mainly five parts to do in this project.
The first is to modify to scanner using yylval like the given example to pass the required attributes to the parser.
Second, modify the hpp files to inherit the functions from the ast file, and specify the member functions and the parameters to use, like the way in the given example.
Third, modify the cpp files to inherit things from the hpp files, and implement the functions, including accept, visitChildNodes, and some functions that we would need to get the parameters in the AstDumper file.
Fourth, modify the dumper file to make the print functions get the parameters we need, and print them out.
Last, but also the most important and difficult one, modify the parser file to specify the things to do when each grammar is met.
The order is not necessarily the order I edited them. In fact, I adjusted them according to the order of the test files.

Next, I would explain how visitor pattern work in my project.
First, the main function would call the root to accept the visitor.
Then, pass the visitor by reference, and AstNode would use the accept function to call visitor.visit(*this), which allows the visitor to visit *this, to meet their true attributes.
After that, the visit functions would be called in the AstDumper file, in which important messages would be printed, and then it would call the child nodes to be visited.
By repeating the above steps, we could parse the whole AST and print all the messages we would like.

## What is the hardest you think in this project

This project is one of the most time-consuming project I've ever written, because there are a lot of difficult parts. 
The first, and also the most difficult I think, is to understand what I had to do in this project. The spec is really really long, and I spent 2 days reading that again and again. After passing the first testing data according to the hint, I still had no idea how I could complete the next parts, and it took me like one day to get the next 9 points. Luckily, after that, I got the feeling and completed the project in the next two days.
Another difficult part is debugging, because there are a lot of files I have to modify, debugging is like finding a needle in the ocean(?). The worst part of debugging is , when I wrote something wrong strangely but "make test" worked, the output became whole blank and I had no ideas what mistakes I made.
The last part I want to mention is that completing this project needs the basis of OOP, and I have forgotten most of that long ago, it took me some time to understand them again. 

## Feedback to T.A.s

This project is really a huge task! It is definitely not easy to make sure that all the things are correct. Also, grading our assignment must also be a nightmare. So, I would like express my gratitude for your dedication. I hope I could do better and better in the remaining semester, and I wish you do well in your graduate life.
