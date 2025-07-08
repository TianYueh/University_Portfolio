classdef lab2_tester
    properties
        golden_data;
    end

    methods
        function obj = lab2_tester()
            obj.golden_data = load('lab2.mat');
        end

        % Q1
        function test1(obj, student_answer, n, r)
            if (n == 7 && r == 2.5) % Task 1: n=7, r=2.5
                golden_answer = obj.golden_data.q1_a;
                task_no = 1;
            elseif (n == 87 && r == 6.3) % Task 2: n=87, r=6.3
                golden_answer = obj.golden_data.q1_b;
                task_no = 2;
            else
                disp("This task does not exist.")
            end
            
            if isequal(golden_answer, student_answer)
                fprintf("Q1 Task%d Pass. \n", task_no)
            else
                fprintf("Q1 Task%d Fail. \n", task_no)
            end
        end
        % Q2
        function test2(obj, student_answer)
            golden_answer = obj.golden_data.q2;
            if isequal(golden_answer, student_answer)
                disp('Q2 Pass.');
            else
                disp('Q2 Fail.');
            end
        end
        % Q3
        function test3(obj, student_answer)
            golden_answer = obj.golden_data.q3;
            if isequal(golden_answer, student_answer)
                disp('Q3 Pass.');
            else
                disp('Q3 Fail.');
            end
        end
    end
end