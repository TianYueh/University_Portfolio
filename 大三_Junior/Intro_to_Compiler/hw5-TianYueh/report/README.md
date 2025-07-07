# hw5 report

|||
|-:|:-|
|Name|房天越|
|ID|110550085|

## How much time did you spend on this project

About 15 hours.

## Project overview

In this project, we implement the code generator for P language to translate it and generate RISC-V assembly languages. The parser would call it to start doing the visitor patterns. When the nodes are being visited, the corresponding assembly codes would be dumped, until all the nodes are visited and all the codes are dumped. We also need some flags to record the status of the generator to dump the corresponding instructions, also the address and the labels of the instructions need to be recorded and dumped.

## What is the hardest you think in this project

With the experiences of the previous projects, this one is relatively simple, and we do not need too much realization about RISC-V to implement it because most of the codes are written in the spec, we only need to understand them and dump them when needed. 
However, there are still a lot of details to consider. The hardest one I think is to generate codes for lvalue and rvalue, we need some flags to record what kind of values they are and generate the corresponding codes for them. It is important to be very careful in this part, or we would end up getting results that we do not want.

## Feedback to T.A.s

So this semester is going to end, and I want to say thank you for giving us assignments and grading our exams and projects, I know this is definitely an exhausting work.
While I did not do well in my exams, I still think that it is a nice experience to participate in the class and do the assignments. I hope that you could do well in your upcoming future.