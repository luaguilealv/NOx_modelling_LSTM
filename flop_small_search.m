clc; clear; close all;
% Matlab functions, validation, data 19 variables + NOx
% Flop optimization using Validation Loss
% Selection of variables

%% Read data
filename = 'var_selec_data.xlsx';
dataTable = readtable(filename); 

y = table2array(dataTable(:,end));
X = table2array(dataTable(:, end-1));

[N, d] = size(X);

%% Chronological train/test-validation split (70/15/15)
ntrain = round(0.7*N);
nval = round(0.15*N);

Xtr = X(1:ntrain,:);
ytr = y(1:ntrain,:);

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

%% Search space (opt. with flops)
T_list = [40 60 70 80 90 100];
H_list = [32 40 64 80 96 110];

%% Hyperparameters
numResponses = 1;
miniBatchSize = 128;

%% FLOP constraint
flopsLimit = 5e6;

%% Tolerance for "good enough" validation loss
epsilon = 0.02;

%% Results 
results = struct( ...
    'T', {}, ...
    'H', {}, ...
    'FLOPs', {}, ...
    'ValLoss', {}, ...
    'ValMSE', {}, ...
    'ValRMSE', {}, ...
    'ValMAE', {}, ...
    'ValR2', {}, ...
    'Net', {});

%% Train and evaluate feasible architectures
for t = 1:length(T_list)

    T = T_list(t);

    % Build sequences
    [Xtr_seq, Ytr_seq] = makeSequences(Xtr_n, ytr_n, T);
    [Xva_seq, Yva_seq] = makeSequences(Xva_n, yva_n, T);

    for h = 1:length(H_list)

        numHidden = H_list(h);

        % FLOP count (LSTM only, consistent with your current formula)
        flops = T * numHidden * (4*d + 4*numHidden + 3);

        % Constraint max flops
        if flops > flopsLimit
            fprintf('Unfeasible: T=%d, H=%d, FLOPs=%.0f > limit %.0f\n', ...
                T, numHidden, flops, flopsLimit);
            continue;
        end

        fprintf('\nTraining candidate: T=%d, H=%d, FLOPs=%.0f\n', ...
            T, numHidden, flops);

        % Define network
        % layers = [
        %     sequenceInputLayer(d)
        %     lstmLayer(numHidden, OutputMode="last")
        %     dropoutLayer(0.2)
        %     fullyConnectedLayer(16)
        %     dropoutLayer(0.2)
        %     fullyConnectedLayer(numResponses)
        % ];
        layers = [
            sequenceInputLayer(d)
            lstmLayer(numHidden, OutputMode="last")
            fullyConnectedLayer(64)
            reluLayer
            fullyConnectedLayer(32)
            reluLayer
            fullyConnectedLayer(numResponses)
        ];

        % layers = [
        %     sequenceInputLayer(d)
        %     lstmLayer(numHidden, OutputMode="last")
        %     fullyConnectedLayer(32)
        %     reluLayer
        %     fullyConnectedLayer(16)
        %     reluLayer
        %     fullyConnectedLayer(numResponses)
        % ];

        % Training options
        options = trainingOptions("adam", ...
            MaxEpochs=50, ...
            MiniBatchSize=miniBatchSize, ...
            InitialLearnRate=1e-3, ...
            ValidationData={Xva_seq, Yva_seq}, ...
            ValidationFrequency=100, ...
            Shuffle="every-epoch", ...
            GradientThreshold=1, ...
            L2Regularization=1e-4, ...
            Verbose=false, ...
            Plots="training-progress");

        % Train network
        [net, info] = trainnet(Xtr_seq, Ytr_seq, layers, "mse", options);

        % Check validation loss
        valLosses = info.ValidationHistory.Loss;
        valLosses = valLosses(~isnan(valLosses));

        if isempty(valLosses)
            fprintf('No valid validation loss for T=%d, H=%d\n', T, numHidden);
            continue;
        end

        % Best validation loss during training
        valLoss = min(valLosses);

        % Predict on validation set
        yhat_va_n = minibatchpredict(net, Xva_seq, MiniBatchSize=miniBatchSize);
        yhat_va_n = yhat_va_n(:);

        % Unstandardize validation predictions and targets
        yhat_va = yhat_va_n .* sigy + muy;
        va_orig = Yva_seq .* sigy + muy;

        % Validation metrics
        mse_va = mean((yhat_va - va_orig).^2);
        rmse_va = sqrt(mse_va);
        mae_va = mean(abs(yhat_va - va_orig));
        R2_va = 1 - sum((va_orig - yhat_va).^2) / sum((va_orig - mean(va_orig)).^2);

        % Print candidate results
        fprintf(['Candidate results: T=%d, H=%d, FLOPs=%.0f, ValLoss=%.6g, ', ...
                 'ValMSE=%.6g, ValRMSE=%.6g, ValMAE=%.6g, ValR2=%.4f\n'], ...
                 T, numHidden, flops, valLoss, mse_va, rmse_va, mae_va, R2_va);

        % Store candidate
        candidate.T = T;
        candidate.H = numHidden;
        candidate.FLOPs = flops;
        candidate.ValLoss = valLoss;
        candidate.ValMSE = mse_va;
        candidate.ValRMSE = rmse_va;
        candidate.ValMAE = mae_va;
        candidate.ValR2 = R2_va;
        candidate.Net = net;

        results(end+1) = candidate;
    end
end

%% Select best architecture using tolerance
if isempty(results)
    error('No feasible architecture was found.');
end

allValLoss = [results.ValLoss];
allFlops   = [results.FLOPs];

% Best validation loss found
minValLoss = min(allValLoss);

% Keep architectures with "good enough" validation loss
goodIdx = allValLoss <= (minValLoss + epsilon);

if ~any(goodIdx)
    error('No architecture satisfies the tolerance criterion.');
end

goodResults = results(goodIdx);

% Among the good-enough architectures, choose the one with minimum FLOPs
[~, idxBest] = min([goodResults.FLOPs]);
bestResult = goodResults(idxBest);

% Save best architecture
bestNet = bestResult.Net;
bestT = bestResult.T;
bestH = bestResult.H;
bestFlops = bestResult.FLOPs;
bestValLoss = bestResult.ValLoss;
bestValMSE = bestResult.ValMSE;
bestValRMSE = bestResult.ValRMSE;
bestValMAE = bestResult.ValMAE;
bestValR2 = bestResult.ValR2;

fprintf('\nTolerance-based selection:\n');
fprintf('Minimum validation loss found = %.6g\n', minValLoss);
fprintf('Tolerance epsilon = %.6g\n', epsilon);
fprintf('Number of good-enough architectures = %d\n', sum(goodIdx));

fprintf('\nBest architecture found:\n');
fprintf('T = %d\n', bestT);
fprintf('numHidden = %d\n', bestH);
fprintf('Approx FLOPs = %.0f\n', bestFlops);
fprintf('Validation loss = %.6g\n', bestValLoss);
fprintf('Validation MSE = %.6g\n', bestValMSE);
fprintf('Validation RMSE = %.6g\n', bestValRMSE);
fprintf('Validation MAE = %.6g\n', bestValMAE);
fprintf('Validation R^2 = %.4f\n', bestValR2);

%% Predict (normalized)
[Xtr_seq, Ytr_seq] = makeSequences(Xtr_n, ytr_n, bestT);
[Xte_seq, Yte_seq] = makeSequences(Xte_n, yte_n, bestT);

yhat_tr_n = minibatchpredict(bestNet, Xtr_seq, MiniBatchSize=miniBatchSize);
yhat_te_n = minibatchpredict(bestNet, Xte_seq, MiniBatchSize=miniBatchSize);

yhat_tr_n = yhat_tr_n(:);
yhat_te_n = yhat_te_n(:);

%% Unstandardize
yhat_tr = yhat_tr_n .* sigy + muy;
yhat_te = yhat_te_n .* sigy + muy;

tr_orig = Ytr_seq .* sigy + muy;
te_orig = Yte_seq .* sigy + muy;

mse_tr = mean((yhat_tr - tr_orig).^2);
mse_te = mean((yhat_te - te_orig).^2);

rmse_te = sqrt(mse_te);
mae_te = mean(abs(yhat_te - te_orig));

R2 = 1 - sum((te_orig - yhat_te).^2) / sum((te_orig - mean(te_orig)).^2);

fprintf('\nFinal performance (original units)\n');
fprintf('Train MSE: %.6g\n', mse_tr);
fprintf('Test  MSE: %.6g\n', mse_te);
fprintf('Test  RMSE: %.6g\n', rmse_te);
fprintf('Test  MAE: %.6g\n', mae_te);
fprintf('Test R^2: %.4f\n', R2);

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

figure;
histogram(yhat_te - te_orig, 20); grid on;
xlabel('Residual (yhat - y)');
title('Test residuals');

figure;
plot(te_orig, 'b'); hold on;
plot(yhat_te, 'r--');
legend('True', 'Predicted');
grid on;
title('Test vs. True NOx');

%% FUNCTIONS
function [Xseq, Yseq] = makeSequences(X, y, T)

    X = double(X);
    y = double(y);

    [N, d] = size(X); 

    M = N - T;   % number of sequences

    Xseq = cell(M,1);
    Yseq = zeros(M,1);

    for i = 1:M
        Xseq{i} = X(i:i+T-1, :);   % [T x d]
        Yseq(i) = y(i+T);          % next-step prediction
    end
end