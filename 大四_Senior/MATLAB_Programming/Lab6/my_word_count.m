function A = my_word_count(fn, sort_mode)
    
    fid = fopen(fn, 'r');
    s = textscan(fid, '%s', 'delimiter', '., ');
    W = lower(s{1}); % convert to lower
    fclose(fid);
    
    % get the unique words
    unique_words = unique(W);
    counts = zeros(size(unique_words));  % allocate for counts by num of unique words

    % count the num of occurrence
    for i = 1:length(unique_words)
        counts(i) = sum(strcmp(W, unique_words{i}));
    end
    
    % create structure with the three attributes
    A = struct('word', [], 'len', [], 'count', []); 
    for i = 1:length(unique_words)
        A(i).word = unique_words{i};  
        A(i).len = length(unique_words{i});  % assign len by the length
        A(i).count = counts(i);  % assign count
    end
    
    % sorting, idx is the index to determine
    switch sort_mode
        case 'word+'
            [~, idx] = sort({A.word});
        case 'word-'
            [~, idx] = sort({A.word});  % sort ascendingly
            idx = flip(idx);  % reverse it, descend can only be applied to scalar
        case 'len+'
            [~, idx] = sort([A.len]);
        case 'len-'
            [~, idx] = sort([A.len], 'descend');
        case 'count+'
            [~, idx] = sort([A.count]);
        case 'count-'
            [~, idx] = sort([A.count], 'descend');
        otherwise
            error('Invalid sort_mode');
    end
    A = A(idx);  % rearrange the array
    
    % if no argout then print
    if nargout == 0
        for i = 1:length(A)
            fprintf('%s: len = %d, count = %d\n', A(i).word, A(i).len, A(i).count);
        end
    end
end
