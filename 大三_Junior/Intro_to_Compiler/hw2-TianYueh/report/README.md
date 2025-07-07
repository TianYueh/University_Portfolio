# hw2 report

|||
|-:|:-|
|Name|房天越|
|ID|110550085|

## How much time did you spend on this project

About 7 hours.

## Project overview

First, I modified the scanner.l file to make it return the keywords or data types that would be used in the parser. Then, I added the rules for the LALR(1) parser to parse according to the spec. So when we have a .p file, we can use scanner.l to recognize the syntax, and then use parser.y to parse it according to the rules I write.

## What is the hardest you think in this project

I think this homework is relatively easy, because what we need to do is modify scanner.l to add the "return"s, and write the rules in scanner.l according to the spec. But there are still some parts that are not that easy.

I would say the hardest part is to understand what I should write in parser.y, just understanding that took me much time. After that, the second hardest part is debugging, realizing why the parser said that there are some unmatched tokens also took me some time, and I found that the mistake is often caused by ignoring a semicolon or some other little mistakes. I also want to mention that handling the two situations, which are "zero or more", and "at least one" are also a little bit challenging.

## Feedback to T.A.s

The homeworks are interesting! I hope I could say this again when handling HW3 and HW4.
I also want to thank you for marking our homeworks and midterm papers. I think I messed up in the midterm, I hope that I could still struggle in this class and do not have to withdraw it QQ.
