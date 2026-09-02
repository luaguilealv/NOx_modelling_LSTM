clc; clear; close all;
% Matlab functions, validation, data 19 variables + NOx

%% Read data
filename = 'var_selec_data.xlsx';
dataTable = readtable(filename); 

X = table2array(dataTable(:,1:end-1));
y = table2array(dataTable(:,end));

%% Plot of NOx but normalized to be able to be included
y_norm = (y - min(y)) / (max(y) - min(y));

figure;
plot(y_norm, 'LineWidth', 1.2);
grid on;
xlabel('Sample index');
ylabel('Normalized NOx');
title('Normalized NOx signal');

%% Correlation Filtering
correlationLimit = 0.4; 

% Calculate correlation of each variable in X with y
% 'Rows','complete' handles any potential NaNs
corr_matrix = corr(X, y, 'Rows', 'complete');

% Find indices where absolute correlation is higher than the limit
selectedIdx = abs(corr_matrix) > correlationLimit;

% Check if we have any variables left!
if ~any(selectedIdx)
    error('No variables meet the correlation threshold. Lower the limit.');
end

% Update X and dimension variable 'd'
X = X(:, selectedIdx);
[N, d] = size(X);

% Display which variables were kept
keptVars = dataTable.Properties.VariableNames(selectedIdx);
fprintf('Variables kept (%d/%d): %s\n', d, size(corr_matrix,1), strjoin(keptVars, ', '));

%% Correlation plot without names so it can be included
absCorr = abs(corr_matrix);

figure;
bar(absCorr);
grid on;
xlabel('Input variable index');
ylabel('|Correlation with target|');
% title('Absolute correlation of inputs with NOx');
yline(correlationLimit, 'r--', 'Threshold', ...
    'LabelHorizontalAlignment', 'left');

%% Chronological train/test-validation split (70/15/15)
ntrain = round(0.7*N);
nval = round(0.15*N);

Xtr = X(1:ntrain,:);
ytr = y(1:ntrain,:);

fprintf('Training target range: [%.3f, %.3f]\n', min(ytr), max(ytr));

Xva = X((ntrain+1):(ntrain+nval),:);
yva = y((ntrain+1):(ntrain+nval),:);

Xte = X((ntrain+nval+1):end,:);
yte = y((ntrain+nval+1):end,:);

%% Standardize (train)
[Xtr_n, muX, sigX] = zscore(Xtr);
sigX(sigX==0) = 1;
Xte_n = (Xte - muX) ./ sigX;
Xva_n = (Xva - muX) ./ sigX;

[ytr_n, muy, sigy] = zscore(ytr);
sigy(sigy==0) = 1;
yte_n = (yte - muy) ./ sigy;
yva_n = (yva - muy) ./ sigy;

%% Build sequences 
T = 80;

[Xtr_seq, Ytr_seq] = makeSequences(Xtr_n, ytr_n, T); % Xtr_seq{i} : [d x T]
[Xte_seq, Yte_seq] = makeSequences(Xte_n, yte_n, T);
[Xva_seq, Yva_seq] = makeSequences(Xva_n, yva_n, T);

Mtr = numel(Xtr_seq);
Mte = numel(Xte_seq);
Mva = numel(Xva_seq);

%% Define network
numHidden    = 80;
numResponses = 1;

layers = [
    sequenceInputLayer(d)
    lstmLayer(numHidden, OutputMode="last")
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(16)
    reluLayer
    fullyConnectedLayer(numResponses)
];

% layers = [
%     sequenceInputLayer(d)
%     lstmLayer(numHidden, OutputMode="last")
%     dropoutLayer(0.2)
%     fullyConnectedLayer(16)
%     dropoutLayer(0.2)
%     fullyConnectedLayer(numResponses)
% ];

%% Training options
miniBatchSize = 128; % 1 for SGD

options = trainingOptions("adam", ...
    MaxEpochs=50, ...
    MiniBatchSize=miniBatchSize, ...
    InitialLearnRate=1e-3, ...
    ValidationData={Xva_seq,Yva_seq}, ...
    ValidationFrequency=100, ...
    Shuffle="every-epoch", ... %"every-epoch"
    GradientThreshold=1, ...
    L2Regularization=1e-4, ...
    Verbose=false, ...
    Plots="training-progress");

%% Train
[net, info] = trainnet(Xtr_seq, Ytr_seq, layers, "mse", options);

%    "mse" | "mae" | "huber"


%% Predict (normalized)
yhat_tr_n = minibatchpredict(net, Xtr_seq, MiniBatchSize=miniBatchSize);
yhat_te_n = minibatchpredict(net, Xte_seq, MiniBatchSize=miniBatchSize);

yhat_tr_n = yhat_tr_n(:);
yhat_te_n = yhat_te_n(:);

%% Unstandardize and evaluate 
yhat_tr = yhat_tr_n .* sigy + muy;
yhat_te = yhat_te_n .* sigy + muy;

tr_orig = Ytr_seq .* sigy + muy;
te_orig = Yte_seq .* sigy + muy;

mse_tr = mean((yhat_tr - tr_orig).^2);
mse_te = mean((yhat_te - te_orig).^2);

fprintf('\nFinal performance (original units)\n');
fprintf('Train MSE: %.6g\n', mse_tr);
fprintf('Test  MSE: %.6g\n', mse_te);

rmse_te = sqrt(mse_te);
fprintf('Test  RMSE: %.6g\n', rmse_te); % In real units, ppm

%% Evaluate 
SSres = sum((te_orig - yhat_te).^2);
SStot = sum((te_orig - mean(te_orig)).^2);
R2 = 1 - SSres/SStot;

fprintf('Test R^2: %.4f\n', R2); % Not enough to make decissions

MAE_te = mean(abs(yhat_te - te_orig));
fprintf("Test MAE: %.3f\n", MAE_te);

%% More Evaluation better than R^2
% Absolute error
abs_err = abs(yhat_te - te_orig);

% 95% of your prediction errors are below this value
P95_error = prctile(abs_err, 95);
fprintf('P95 absolute error = %.4f\n', P95_error);

% Maximum absolute error anywhere
peak_error_global = max(abs_err);
fprintf('Maximum absolute error = %.4f\n', peak_error_global);

% Error at the true NOx peak
[true_peak_value, idx_true_peak] = max(te_orig);
pred_at_true_peak = yhat_te(idx_true_peak);
peak_error_at_true_peak = abs(true_peak_value - pred_at_true_peak);

fprintf('Error at true peak = %.4f\n', peak_error_at_true_peak);

%% Plots
figure;
subplot(1,2,1);
scatter(tr_orig, yhat_tr, 'filled'); grid on;
xlabel('True'); ylabel('Pred'); title('Train: true vs pred');
hold on;
mn = min([tr_orig; yhat_tr]);
mx = max([tr_orig; yhat_tr]);
plot([mn mx], [mn mx], 'k--', 'LineWidth', 1.2);
hold off;

subplot(1,2,2);
scatter(te_orig, yhat_te, 'filled'); grid on;
xlabel('True'); ylabel('Pred'); title('Test: true vs pred');
hold on;
mn = min([te_orig; yhat_te]);
mx = max([te_orig; yhat_te]);
plot([mn mx], [mn mx], 'k--', 'LineWidth', 1.2);
hold off;

%% Save trained model and preprocessing parameters
modelFile = 'model.mat';

save(modelFile, ...
    'net',...
    'muX', 'sigX', ...
    'muy', 'sigy', ...
    'selectedIdx', ...
    'keptVars', ...
    'T', ...
    'correlationLimit');

fprintf('Model saved to %s\n', modelFile);

%% FUNCTIONS 

function [Xseq, Yseq] = makeSequences(X, y, T)

    X = double(X);
    y = double(y);

    [N,~] = size(X);

    M = N - T;   % number of sequences

    Xseq = cell(M,1);
    Yseq = zeros(M,1);

    for i = 1:M
        Xseq{i} = X(i:i+T-1, :);   % [T x d] because trainnet needs to have it in this order
        Yseq(i) = y(i+T);           % next-step prediction
    end
end