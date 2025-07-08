IM = rgb2gray(imread('input.jpg'));

figure; 
imshow(IM);

histogram = zeros(1, 256);

for pixel_value = 0:255
    histogram(pixel_value + 1) = sum(IM(:) == pixel_value);
end

sumup = cumsum(histogram);
rs = sumup(256);

H = zeros(1 ,256);
H = histogram./rs;

w = zeros(1, 256);
w = uint8(cumsum(H)*255);

% Need +1 because index from 1 to 256
im_new = w(double(IM) + 1); 

figure;
imshow(im_new);


tester = lab2_tester();
tester.test3(im_new);