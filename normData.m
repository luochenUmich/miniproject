function normData = normData(data)
numRows = size(data, 1);
normdata = data;
for i = 1:numRows
    norm10 = norm(data(i, :)) / 10;
    normData(i, :) = data(i, :) / norm10;
end
end