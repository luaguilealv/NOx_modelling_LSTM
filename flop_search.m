clc; clear; close all;
% Matlab functions, validation, data 19 variables + NOx
% Flop optimization using Validation Loss
% Selection of variables

%% Read data
filename = 'var_selec_data.xlsx';
dataTable = readtable(filename); 

X = table2array(dataTable(:,1:end-1));
y = table2array(dataTable(:,end));

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

%% Standardize using training data only
[Xtr_n, muX, sigX] = zscore(Xtr);
sigX(sigX==0) = 1;

Xva_n = (Xva - muX) ./ sigX;
Xte_n = (Xte - muX) ./ sigX;

[ytr_n, muy, sigy] = zscore(ytr);
sigy(sigy==0) = 1;

yva_n = (yva - muy) ./ sigy;
yte_n = (yte - muy) ./ sigy;

%% Search space
T_list = [20 50 70 80 100 150];
H_list = [10 16 20 32 40 64 80 96 110 120];

%% Hyperparameters
numResponses = 1;
miniBatchSize = 128;
flopsLimit = 5e6;
epsilon = 0.02;

%% Architectures and loss functions
architectureNames = ["Dropout_FC16", "FC32_FC16_ReLU"];
lossFunctions = ["mse", "mae", "huber"];

%% Store global summary
allBestModels = struct([]);

counter = 0;

%% Loop over loss functions
for lf = 1:length(lossFunctions)

    lossName = lossFunctions(lf);

    %% Loop over architectures
    for arch = 1:length(architectureNames)

        archName = architectureNames(arch);

        fprintf('\n========================================\n');
        fprintf('Loss function: %s | Architecture: %s\n', lossName, archName);
        fprintf('========================================\n');

        results = struct( ...
            'LossFunction', {}, ...
            'Architecture', {}, ...
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

            [Xtr_seq, Ytr_seq] = makeSequences(Xtr_n, ytr_n, T);
            [Xva_seq, Yva_seq] = makeSequences(Xva_n, yva_n, T);

            for h = 1:length(H_list)

                numHidden = H_list(h);

                %% FLOP count
                flops = T * numHidden * (4*d + 4*numHidden + 3);

                if flops > flopsLimit
                    fprintf('Unfeasible: Loss=%s, Arch=%s, T=%d, H=%d, FLOPs=%.0f > limit %.0f\n', ...
                        lossName, archName, T, numHidden, flops, flopsLimit);
                    continue;
                end

                fprintf('\nTraining candidate: Loss=%s, Arch=%s, T=%d, H=%d, FLOPs=%.0f\n', ...
                    lossName, archName, T, numHidden, flops);

                %% Define architecture
                if archName == "Dropout_FC16"

                    layers = [
                        sequenceInputLayer(d)
                        lstmLayer(numHidden, OutputMode="last")
                        dropoutLayer(0.2)
                        fullyConnectedLayer(16)
                        dropoutLayer(0.2)
                        fullyConnectedLayer(numResponses)
                    ];

                elseif archName == "FC32_FC16_ReLU"

                    layers = [
                        sequenceInputLayer(d)
                        lstmLayer(numHidden, OutputMode="last")
                        fullyConnectedLayer(32)
                        reluLayer
                        fullyConnectedLayer(16)
                        reluLayer
                        fullyConnectedLayer(numResponses)
                    ];
                end

                %% Training options
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
                    ExecutionEnvironment="gpu",...
                    Plots="training-progress");

                %% Train network using current loss function
                [net, info] = trainnet(Xtr_seq, Ytr_seq, layers, lossName, options);

                %% Validation loss from training history
                valLosses = info.ValidationHistory.Loss;
                valLosses = valLosses(~isnan(valLosses));

                if isempty(valLosses)
                    fprintf('No valid validation loss for Loss=%s, Arch=%s, T=%d, H=%d\n', ...
                        lossName, archName, T, numHidden);
                    continue;
                end

                valLoss = min(valLosses);

                %% Validation prediction
                yhat_va_n = minibatchpredict(net, Xva_seq, MiniBatchSize=miniBatchSize);
                yhat_va_n = yhat_va_n(:);

                yhat_va = yhat_va_n .* sigy + muy;
                va_orig = Yva_seq .* sigy + muy;

                %% Validation metrics in original units
                mse_va = mean((yhat_va - va_orig).^2);
                rmse_va = sqrt(mse_va);
                mae_va = mean(abs(yhat_va - va_orig));
                R2_va = 1 - sum((va_orig - yhat_va).^2) / sum((va_orig - mean(va_orig)).^2);

                fprintf(['Candidate results: Loss=%s, Arch=%s, T=%d, H=%d, FLOPs=%.0f, ', ...
                         'ValLoss=%.6g, ValMSE=%.6g, ValRMSE=%.6g, ', ...
                         'ValMAE=%.6g, ValR2=%.4f\n'], ...
                         lossName, archName, T, numHidden, flops, valLoss, ...
                         mse_va, rmse_va, mae_va, R2_va);

                %% Store candidate
                candidate.LossFunction = lossName;
                candidate.Architecture = archName;
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

        %% Tolerance-based selection for this loss + architecture
        if isempty(results)
            warning('No feasible model found for Loss=%s, Architecture=%s', lossName, archName);
            continue;
        end

        allValLoss = [results.ValLoss];
        minValLoss = min(allValLoss);

        goodIdx = allValLoss <= (minValLoss + epsilon);
        goodResults = results(goodIdx);

        [~, idxBest] = min([goodResults.FLOPs]);
        bestResult = goodResults(idxBest);

        bestNet = bestResult.Net;
        bestT = bestResult.T;
        bestH = bestResult.H;

        %% Test evaluation for selected model
        [Xtr_seq, Ytr_seq] = makeSequences(Xtr_n, ytr_n, bestT);
        [Xte_seq, Yte_seq] = makeSequences(Xte_n, yte_n, bestT);

        yhat_tr_n = minibatchpredict(bestNet, Xtr_seq, MiniBatchSize=miniBatchSize);
        yhat_te_n = minibatchpredict(bestNet, Xte_seq, MiniBatchSize=miniBatchSize);

        yhat_tr_n = yhat_tr_n(:);
        yhat_te_n = yhat_te_n(:);

        yhat_tr = yhat_tr_n .* sigy + muy;
        yhat_te = yhat_te_n .* sigy + muy;

        tr_orig = Ytr_seq .* sigy + muy;
        te_orig = Yte_seq .* sigy + muy;

        mse_tr = mean((yhat_tr - tr_orig).^2);
        mse_te = mean((yhat_te - te_orig).^2);
        rmse_te = sqrt(mse_te);
        mae_te = mean(abs(yhat_te - te_orig));

        R2_te = 1 - sum((te_orig - yhat_te).^2) / sum((te_orig - mean(te_orig)).^2);

        abs_err = abs(yhat_te - te_orig);
        P95_error = prctile(abs_err, 95);
        peak_error_global = max(abs_err);

        [true_peak_value, idx_true_peak] = max(te_orig);
        pred_at_true_peak = yhat_te(idx_true_peak);
        peak_error_at_true_peak = abs(true_peak_value - pred_at_true_peak);

        %% Print best result
        fprintf('\nBest model for Loss=%s | Architecture=%s\n', lossName, archName);
        fprintf('T = %d\n', bestT);
        fprintf('H = %d\n', bestH);
        fprintf('FLOPs = %.0f\n', bestResult.FLOPs);
        fprintf('Validation loss = %.6g\n', bestResult.ValLoss);
        fprintf('Validation MSE = %.6g\n', bestResult.ValMSE);
        fprintf('Validation RMSE = %.6g\n', bestResult.ValRMSE);
        fprintf('Validation MAE = %.6g\n', bestResult.ValMAE);
        fprintf('Validation R^2 = %.4f\n', bestResult.ValR2);

        fprintf('\nTest performance:\n');
        fprintf('Train MSE = %.6g\n', mse_tr);
        fprintf('Test MSE = %.6g\n', mse_te);
        fprintf('Test RMSE = %.6g\n', rmse_te);
        fprintf('Test MAE = %.6g\n', mae_te);
        fprintf('Test R^2 = %.4f\n', R2_te);
        fprintf('P95 absolute error = %.4f\n', P95_error);
        fprintf('Maximum absolute error = %.4f\n', peak_error_global);
        fprintf('Error at true peak = %.4f\n', peak_error_at_true_peak);

        %% Save best model
        saveName = sprintf('NOx_LSTM_best_%s_%s.mat', lossName, archName);

        save(saveName, ...
            'bestNet', 'bestT', 'bestH', 'bestResult', ...
            'lossName', 'archName', ...
            'muX', 'sigX', 'muy', 'sigy', ...
            'var', 'd', 'miniBatchSize', ...
            'mse_tr', 'mse_te', 'rmse_te', 'mae_te', 'R2_te', ...
            'P95_error', 'peak_error_global', 'peak_error_at_true_peak');

        fprintf('Saved best model: %s\n', saveName);

        %% Store global summary
        counter = counter + 1;

        allBestModels(counter).LossFunction = lossName;
        allBestModels(counter).Architecture = archName;
        allBestModels(counter).T = bestT;
        allBestModels(counter).H = bestH;
        allBestModels(counter).FLOPs = bestResult.FLOPs;
        allBestModels(counter).ValLoss = bestResult.ValLoss;
        allBestModels(counter).ValMSE = bestResult.ValMSE;
        allBestModels(counter).ValRMSE = bestResult.ValRMSE;
        allBestModels(counter).ValMAE = bestResult.ValMAE;
        allBestModels(counter).ValR2 = bestResult.ValR2;
        allBestModels(counter).TestMSE = mse_te;
        allBestModels(counter).TestRMSE = rmse_te;
        allBestModels(counter).TestMAE = mae_te;
        allBestModels(counter).TestR2 = R2_te;
        allBestModels(counter).P95Error = P95_error;
        allBestModels(counter).MaxError = peak_error_global;
        allBestModels(counter).PeakError = peak_error_at_true_peak;
    end
end

%% Save summary table
summaryTable = struct2table(allBestModels);
disp(summaryTable);

writetable(summaryTable, 'best_models_by_loss_and_architecture.xlsx');

%% FUNCTIONS
function [Xseq, Yseq] = makeSequences(X, y, T)

    X = double(X);
    y = double(y);

    N = size(X,1);
    M = N - T;

    Xseq = cell(M,1);
    Yseq = zeros(M,1);

    for i = 1:M
        Xseq{i} = X(i:i+T-1, :);
        Yseq(i) = y(i+T);
    end
end