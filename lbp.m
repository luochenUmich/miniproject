function lbpFeatures = lbp(data)
lbpFeatureDimension = (48/8)*(48/8)*58;
numRows = size(data, 1);
numWindow = 48/8;
lbpFeatures = zeros(numRows, lbpFeatureDimension);
for i = 1:numRows
     HOG = vl_lbp(single(vec2mat(data(i, :), 48))', 8);
     feature = [];
     for row = 1:numWindow
         for col = 1:numWindow
             feature = [feature reshape(HOG(row, col, :), [1, 58])];
         end
     end
     lbpFeatures(i, :) = feature;
end
end