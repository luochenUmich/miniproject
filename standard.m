function data = standard(data)
numCols = size(data, 2);
for col = 1:numCols
    data(:, col) = data(:, col) - mean(data(:, col));
    data(:, col) = data(:, col) ./ std(data(:, col));
end
end