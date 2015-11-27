function softmaxModel = softmaxTrain(inputData, labels, opts)
% softmaxModel = softmaxTrain(inputData, labels, opts)
% Train a softmax model with the given parameters on the given data.
% Returns softmaxOptTheta, a vector containing the trained parameters
% for the model.
% Arguments:
% inputData - an N by M matrix containing the input data, such that
%            inputData(:, c) is the cth input
% labels - M by 1 matrix containing the class labels (marked from 1) for the
%            corresponding inputs. labels(c) is the class label for
%            the cth input
% options (optional) - opts
%   opts.maxIter - number of iterations to train for
%   opts.lambda - weight decay parameter

if ~exist('options', 'var'), opts = struct; end
if ~isfield(opts, 'maxIter'), opts.maxIter = 500; end
if ~isfield(opts, 'lambda'), opts.lambda = 1e-3; end

numClasses = numel(unique(labels));
inputSize = size(inputData, 1);

% initialize parameters
theta = 0.005 * randn(numClasses * inputSize, 1);

% Use minFunc to minimize the function
opts.Method = 'lbfgs';
% Here, we use L-BFGS to optimize our cost
% function. Generally, for minFunc to work, you
% need a function pointer with two outputs: the
% function value and the gradient. In our problem,
% softmaxCost.m satisfies this.

[softmaxOptTheta, cost] = minFunc( @(p) softmaxCost(p, ...
    numClasses, inputSize, opts.lambda, inputData, labels), ...
    theta, opts);

% Fold softmaxOptTheta into a nicer format
softmaxModel.optTheta = reshape(softmaxOptTheta, numClasses, inputSize);
softmaxModel.inputSize = inputSize;
softmaxModel.numClasses = numClasses;

end
