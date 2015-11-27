function initConvnetModel = getNeuralNet(activation, scale)
% The initial value for weights and biases in convolutional layers
% In this mini project, we just specify them for you.
% You will need to find out the optimal parameters by yourself in
% real-world practice.
initW = 1e-2;
initB = 1e-1;

initConvnetModel.layers = {};
% Layer 1: Convolutional Layer
initConvnetModel.layers{end+1} = struct('type', 'conv',...
    'weights', {{initW * randn(5,5,1,ceil(32 * scale), 'single'),...
    initB*randn(1,ceil(32 * scale), 'single')}},...
    'stride', 1,...
    'pad', 2);

% Layer 2: Activation Function
if strcmp(activation, 'none') == 0
    initConvnetModel.layers{end+1} = struct('type', activation);
end

% Layer 3: Pooling Layer
initConvnetModel.layers{end+1} = struct('type', 'pool',...
    'method', 'max',...
    'pool', [3, 3],...
    'stride',2,...
    'pad', 1);

% Layer 4: Convolutional Layer
initConvnetModel.layers{end+1} = struct('type', 'conv',...
    'weights', {{initW * randn(5,5,ceil(32 * scale),ceil(64 * scale), 'single'),...
    initB * randn(1, ceil(64 * scale), 'single')}}, ...
    'stride', 1,...
    'pad',2);

% Layer 5: Activation Function
if strcmp(activation, 'none') == 0
    initConvnetModel.layers{end+1} = struct('type', activation);
end

% Layer 6: Pooling Layer
initConvnetModel.layers{end+1} = struct('type', 'pool',...
    'method', 'max',...
    'pool', [3, 3],...
    'stride',2,...
    'pad', 1);

% Layer 7: Convolutional Layer
initConvnetModel.layers{end+1} = struct('type', 'conv',...
    'weights', {{initW * randn(8,8,ceil(64 * scale), ceil(2048 * scale), 'single'),...
    initB * randn(1, ceil(2048 * scale), 'single')}}, ...
    'stride', 1,...
    'pad',0);

% Layer 8: Activation Function
if strcmp(activation, 'none') == 0
    initConvnetModel.layers{end+1} = struct('type', activation);
end

% Layer 9: Convolutional Layer
initConvnetModel.layers{end+1} = struct('type', 'conv',...
    'weights', {{initW * randn(1,1,ceil(2048 * scale),7, 'single'),...
    initB * randn(1, 7, 'single')}}, ...
    'stride', 1,...
    'pad',0);

end