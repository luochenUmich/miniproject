function resizedData = downscale(data)
numRows = size(data, 1);
originSize = 48;
newSize = 32;
ratio = newSize / originSize;
resizedData = zeros(numRows, newSize * newSize);
for i = 1:numRows
    figure = imresize(vec2mat(data(i, :), originSize)', ratio);
    resizedData(i, :) = reshape(figure, 1, newSize * newSize);
end
end