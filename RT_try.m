%clc; clear; close all;

%% Load trained model
S = load('NOx_model_1');

net = S.net;
muX = S.muX;
sigX = S.sigX;
muy = S.muy;
sigy = S.sigy;
selectedIdx = S.selectedIdx;
keptVars = S.keptVars;
T = S.T;

%% Load new dataset
filename = 'var_selec_data3.xlsx';   
dataTable = readtable(filename);

targetName = 'NOX';        

%% Separate inputs and target
X_raw = table2array(dataTable(:,1:end-1));
y_raw = table2array(dataTable(:,targetName));

%% Same feature selection 
X_sel = X_raw(:, selectedIdx);
disp('Variables expected by the model:');
disp(keptVars');

%% Standardize using training 
X_n = (X_sel - muX) ./ sigX;
y_n = (y_raw - muy) ./ sigy;

%% Build sequences 
[X_seq, Y_seq] = makeSequences(X_n, y_n, T);

%% Predict
miniBatchSize = 128;  
yhat_n = minibatchpredict(net, X_seq, MiniBatchSize=miniBatchSize);
yhat_n = yhat_n(:);

%% Convert predictions back to original units
yhat = yhat_n .* sigy + muy;
ytrue = Y_seq .* sigy + muy;

%% Metrics
mse_val = mean((yhat - ytrue).^2);
rmse_val = sqrt(mse_val);
mae_val = mean(abs(yhat - ytrue));

SSres = sum((ytrue - yhat).^2);
SStot = sum((ytrue - mean(ytrue)).^2);
R2 = 1 - SSres/SStot;

abs_err = abs(yhat - ytrue);
P95_error = prctile(abs_err, 95);
max_error = max(abs_err);

fprintf('\nPerformance on new dataset\n');
fprintf('MSE   : %.6g\n', mse_val);
fprintf('RMSE  : %.6g\n', rmse_val);
fprintf('MAE   : %.6g\n', mae_val);
fprintf('R^2   : %.4f\n', R2);
fprintf('P95 AE: %.4f\n', P95_error);
fprintf('Max AE: %.4f\n', max_error);

%% Plot true vs predicted
figure;
plot(ytrue, 'b', 'LineWidth', 1.2); hold on;
plot(yhat, 'r--', 'LineWidth', 1.2);
grid on;
xlabel('Sequence index');
ylabel('NOx');
legend('True','Predicted');
%title('NOx prediction on new dataset');

%% Scatter plot
figure;
scatter(ytrue, yhat, 14, 'filled');
grid on;
xlabel('True NOx');
ylabel('Predicted NOx');
%title('Predicted vs True');

hold on;
mn = min([ytrue; yhat]);
mx = max([ytrue; yhat]);
plot([mn mx], [mn mx], 'k--', 'LineWidth', 1.2);
hold off;

%% Residual histogram
residuals = yhat - ytrue;

figure;
histogram(residuals, 25);
grid on;
xlabel('Residual');
ylabel('Frequency');
%title('Residual distribution');

%% Plot true vs predicted - normalized data
figure;
plot(Y_seq, 'b', 'LineWidth', 1.2); hold on;
plot(yhat_n, 'r--', 'LineWidth', 1.2);
grid on;
xlabel('Sequence index');
ylabel('NOx');
legend('True','Predicted');
%title('Normalized NOx prediction on new dataset');

%% Scatter plot - normalized data
figure;
scatter(Y_seq, yhat_n, 14, 'filled');
grid on;
xlabel('True NOx');
ylabel('Predicted NOx');
%title('Predicted vs True - normalized data');

hold on;
mn_n = min([Y_seq; yhat_n]);
mx_n = max([Y_seq; yhat_n]);
plot([mn_n mx_n], [mn_n mx_n], 'k--', 'LineWidth', 1.2);
hold off;

%% Residual histogram - normalized data
residuals_n = yhat_n - Y_seq;

figure;
histogram(residuals_n, 25);
grid on;
xlabel('Normalized residual');
ylabel('Frequency');
%title('Residual distribution - normalized data');

%% Function
function [Xseq, Yseq] = makeSequences(X, y, T)

    X = double(X);
    y = double(y);

    [N,d] = size(X);

    M = N - T;   % number of sequences

    Xseq = cell(M,1);
    Yseq = zeros(M,1);

    for i = 1:M
        Xseq{i} = X(i:i+T-1, :);   % [T x d] because trainnet needs to have it in this order
        Yseq(i) = y(i+T);           % next-step prediction
    end
end