IM = rgb2gray(imread('input.jpg'));

figure; 
imshow(IM);

histogram = zeros(1, 256);

for pixel_value = 0:255
    histogram(pixel_value + 1) = sum(IM(:) == pixel_value);
end

histogram;

histogram_builtin = imhist(IM, 256)
histogram_builtin = histogram_builtin'


%dif = zeros(1, 256);

%dif = histogram_builtin-histogram_manual;


figure;
hold on;
plot(0:255, histogram, 'k', 'Displayname', 'Manual');
plot(0:255, histogram_builtin, 'r--', 'Displayname', 'Builtin');
title('Comparison');
xlabel('PixelValue');
ylabel('NumOfPixels');
legend;
hold off;



tester = lab2_tester();
tester.test2(histogram);