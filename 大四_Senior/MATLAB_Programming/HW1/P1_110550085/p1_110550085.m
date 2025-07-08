%110550085房天越

% Load the images
A = imread('input1_110550085.jpg'); % Base image
B = imread('input2_110550085.jpg'); % Overlay image

% Create the first figure for base image
figure('Name', 'Figure 1'); 
imshow(A);                  
title('Base Image (A)');

% Create the second figure for overlay image
figure('Name', 'Figure 2'); 
imshow(B);                  
title('Overlay Image (B)');

% Resize the overlay image (B) to a smaller size to fit into A
% Resize overlay image to 580*580 pixels
overlay_size = [580, 580]; 
B = imresize(B, overlay_size);

% Define the position where to overlay B onto A
position_x = 100;
position_y = 130;

% Create a mask for B where the background will be excluded
% Need two thresholds to distinguish green and white, white needs to be reserved
green_threshold = 150; 
blue_threshold = 100;
% Range determined by using B(:, :, c) to check the values
mask = (B(:,:,2) < green_threshold) | (B(:, :, 3) > blue_threshold); 

% Copy the non-masked pixels from B to A for the three channels
% 1 to 3 corresponds to RGB
for c = 1:3
    % Extract the channel for both A and B
    A_channel = A(position_y:(position_y + size(B, 1) - 1), position_x:(position_x + size(B, 2) - 1), c);
    B_channel = B(:,:,c);
    
    % Only copy the pixels where the mask is true
    A_channel(mask) = B_channel(mask);
    
    % Replace the part of A with the new blended channel
    A(position_y:(position_y + size(B, 1) - 1), position_x:(position_x + size(B, 2) - 1), c) = A_channel;
end

% Display the result
figure('Name', 'Figure 3'); 
imshow(A);                  
title('Final Image (A)');

% Save the result image A_result as a JPEG file
imwrite(A, 'output_110550085.jpg');

