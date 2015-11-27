function data = normData(data)
numRows = size(data, 1);
for i = 1:numRows
    norm10 = norm(data(i, :)) / 10;
    data(i, :) = data(i, :) / norm10;
end
end