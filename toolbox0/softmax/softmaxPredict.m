function pred = softmaxPredict(softmaxModel, data)
% pred = softmaxPredict(softmaxModel, data)
% Make predictions on data using the Softmax model. Return pred, a row vector
% where pred(i) is argmax_c P(y(c) | x(i)).
% Arguments:
% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

P = theta * data;
P = bsxfun(@minus, P, max(P, [], 1));
P = exp(P);
P = bsxfun(@rdivide, P, sum(P));

[~, pred] = max(P, [], 1);

end

