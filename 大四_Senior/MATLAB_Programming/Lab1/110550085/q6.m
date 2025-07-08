a = 1:4;
b = 2:6;
b = b';
lena = length(a);
lenb = length(b);
%[X, Y] = meshgrid(a, b)
X = repmat(a, lenb, 1)
Y = repmat(b, 1, lena)
