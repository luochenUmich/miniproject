function initConvnetModel = getBiggerNeuralNet()
initW = 1e-2;
initB = 1e-1;

initConvnetModel.layers = {};
% Layer 1: Convolutional Layer
initConvnetModel.layers{end+1} = struct('type', 'conv',...
    'weights', {{initW * randn(5,5,1,32, 'single'),...
    initB*randn(1,32, 'single')}},...
    'stride', 1,...
    'pad', 2);

% Layer 2: Activation Function
initConvnetModel.layers{end+1} = struct('type', 'relu');

% Layer 3: Pooling Layer
initConvnetModel.layers{end+1} = struct('type', 'pool',...
    'method', 'max',...
    'pool', [3, 3],...
    'stride',2,...
    'pad', 1);

% Layer 4: Convolutional Layer
initConvnetModel.layers{end+1} = struct('type', 'conv',...
    'weights', {{initW * randn(5,5,32,64, 'single'),...
    initB * randn(1,64, 'single')}}, ...
    'stride', 1,...
    'pad',2);

% Layer 5: Activation Function
initConvnetModel.layers{end+1} = struct('type', 'relu');

% Layer 6: Pooling Layer
initConvnetModel.layers{end+1} = struct('type', 'pool',...
    'method', 'max',...
    'pool', [3, 3],...
    'stride',2,...
    'pad', 1);

% Layer 7: Convolutional Layer
initConvnetModel.layers{end+1} = struct('type', 'conv',...
    'weights', {{initW * randn(5,5,64,64, 'single'),...
    initB * randn(1,64, 'single')}}, ...
    'stride', 1,...
    'pad',2);

% Layer 8: Activation Function
initConvnetModel.layers{end+1} = struct('type', 'relu');

% Layer 9: Pooling Layer
initConvnetModel.layers{end+1} = struct('type', 'pool',...
    'method', 'max',...
    'pool', [3, 3],...
    'stride',2,...
    'pad', 1);

% Layer 10: Convolutional Layer
initConvnetModel.layers{end+1} = struct('type', 'conv',...
    'weights', {{initW * randn(6,6,64,4096, 'single'),...
    initB * randn(1, 4096, 'single')}}, ...
    'stride', 1,...
    'pad',0);

% Layer 11: Activation Function
initConvnetModel.layers{end+1} = struct('type', 'relu');

% Layer 12: Convolutional Layer
initConvnetModel.layers{end+1} = struct('type', 'conv',...
    'weights', {{initW * randn(1,1,4096,7, 'single'),...
    initB * randn(1, 7, 'single')}}, ...
    'stride', 1,...
    'pad',0);

% Layer 13: Softmax Loss Layer
initConvnetModel.layers{end+1} = struct('type', 'softmaxloss');

end