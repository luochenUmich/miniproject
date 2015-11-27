function hogFeatures = hog(data)
hogFeatureDimension = (48/8)*(48/8)*31;
numRows = size(data, 1);
numWindow = 48/8;
hogFeatures = zeros(numRows, hogFeatureDimension);
for i = 1:numRows
     HOG = vl_hog(single(vec2mat(data(i, :), 48))', 8);
     feature = [];
     for row = 1:numWindow
         for col = 1:numWindow
             feature = [feature reshape(HOG(row, col, :), [1, 31])];
         end
     end
     hogFeatures(i, :) = feature;
end
end