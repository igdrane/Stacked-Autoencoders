%% Optimisation de l'architecture AE1+AE2+Softmax avec fine-tuning
% H1 est fixé (valeur optimale trouvée précédemment), H2 varie.
%
% Architecture testée pour chaque H2 :
%   AE1(H1fixe) + AE2(H2) + Softmax  (avec fine-tuning global)
%
% Objectif :
%   - voir jusqu'à quel point on peut réduire/augmenter H2 tout en gardant
%     une précision acceptable (~99 %),
%   - comparer ensuite aux résultats du réseau de référence H1=60, H2=50.

clear; close all; clc;
rng('default');  % reproductibilité

%% 0. Paramètres généraux

resultsDir = 'results_opt_H2_AE1AE2_FT';
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

% ---- VALEURS À TESTER ----
H1 = 60;                          % encodeur 1 fixé (valeur optimale trouvée avant)
H2_list = [10 20 30 40];          % encodeur 2 : valeurs à tester
% ---------------------------

maxEpochsAE   = 100;
L2reg         = 0.001;
sparsityReg   = 4;
sparsityProp  = 0.15;

fprintf('================= PARAMÈTRES GÉNÉRAUX =================\n');
fprintf('H1 (AE1) fixé                 = %d neurones\n', H1);
fprintf('H2 testé                      = ['); fprintf('%d ', H2_list); fprintf(']\n');
fprintf('MaxEpochs (AE)                = %d\n', maxEpochsAE);
fprintf('L2                            = %.4f\n', L2reg);
fprintf('SparsityReg                   = %.2f\n', sparsityReg);
fprintf('SparsityProp                  = %.2f\n', sparsityProp);
fprintf('========================================================\n\n');

%% 1. Chargement et préparation des données

[xTrainImages, tTrain] = digitTrainCellArrayData;
[xTestImages,  tTest]  = digitTestCellArrayData;

numTrain = numel(xTrainImages);
numTest  = numel(xTestImages);
fprintf('Nombre d''images d''apprentissage : %d\n', numTrain);
fprintf('Nombre d''images de test          : %d\n\n', numTest);

% Vectorisation pour le fine-tuning
xTrain = vectorizeImages(xTrainImages);  % 784 x Ntrain
xTest  = vectorizeImages(xTestImages);   % 784 x Ntest

%% 2. Boucle sur les différentes valeurs de H2

nConfigs = numel(H2_list);
acc_all      = zeros(1, nConfigs);
err_all      = zeros(1, nConfigs);
nParams_all  = zeros(1, nConfigs);
classAcc_all = zeros(nConfigs, 10);  % 10 classes

for k = 1:nConfigs
    H2 = H2_list(k);
    fprintf('\n============== Configuration %d / %d : H1 = %d, H2 = %d ==============\n', ...
        k, nConfigs, H1, H2);

    %% 2.1 AE1 sur images (H1 fixé)
    fprintf('Entraînement AE1 (H1 = %d)\n', H1);
    autoenc1 = trainAutoencoder( ...
        xTrainImages, ...
        H1, ...
        'MaxEpochs',              maxEpochsAE, ...
        'L2WeightRegularization', L2reg, ...
        'SparsityRegularization', sparsityReg, ...
        'SparsityProportion',     sparsityProp, ...
        'ScaleData',              false);

    feat1Train = encode(autoenc1, xTrainImages);  % dimension H1

    %% 2.2 AE2 sur les features de AE1 (taille H2 variable)
    fprintf('Entraînement AE2 (entrée = H1 = %d, H2 = %d)\n', H1, H2);
    autoenc2 = trainAutoencoder( ...
        feat1Train, ...
        H2, ...
        'MaxEpochs',              maxEpochsAE, ...
        'L2WeightRegularization', L2reg, ...
        'SparsityRegularization', sparsityReg, ...
        'SparsityProportion',     sparsityProp, ...
        'ScaleData',              false);

    feat2Train = encode(autoenc2, feat1Train);  % dimension H2

    %% 2.3 Softmax sur les features de AE2
    fprintf('Entraînement de la couche softmax\n');
    softnet = trainSoftmaxLayer(feat2Train, tTrain, 'MaxEpochs', 100);

    %% 2.4 Réseau empilé + fine-tuning global
    fprintf('Fine-tuning global du réseau AE1+AE2+Softmax\n');
    net = stack(autoenc1, autoenc2, softnet);   % empilement
    net_FT = train(net, xTrain, tTrain);        % fine-tuning global

    %% 2.5 Évaluation sur la base de test
    figPrefix = sprintf('opt_H1_%d_H2_%d_FT', H1, H2);
    [acc, err, nParams, classAcc] = evaluateNetwork( ...
        net_FT, xTest, tTest, figPrefix, resultsDir);

    acc_all(k)      = acc;
    err_all(k)      = err;
    nParams_all(k)  = nParams;
    classAcc_all(k, :) = classAcc;

    fprintf('\n--> Résultats pour H1 = %d, H2 = %d :\n', H1, H2);
    fprintf('Accuracy globale = %.2f %% | Erreur = %.2f %% | Nb paramètres = %d\n', ...
        acc*100, err*100, nParams);
    fprintf('Accuracy par classe (1..10) = ['); fprintf(' %.2f', classAcc*100); fprintf(' ] %%\n');
end

%% 3. Courbes de synthèse (H2 vs accuracy, erreur, nb paramètres)

% Accuracy
figure;
plot(H2_list, acc_all*100, '-o', 'LineWidth', 1.5);
xlabel('Nombre de neurones H2 (AE2)');
ylabel('Accuracy globale (%)');
title(sprintf('AE1(H1=%d)+AE2+Softmax avec FT - Accuracy vs H2', H1));
grid on;
saveas(gcf, fullfile(resultsDir, 'opt_accuracy_vs_H2.png'));

% Erreur
figure;
plot(H2_list, err_all*100, '-o', 'LineWidth', 1.5);
xlabel('Nombre de neurones H2 (AE2)');
ylabel('Erreur de classification (%)');
title(sprintf('AE1(H1=%d)+AE2+Softmax avec FT - Erreur vs H2', H1));
grid on;
saveas(gcf, fullfile(resultsDir, 'opt_error_vs_H2.png'));

% Nombre de paramètres
figure;
plot(H2_list, nParams_all, '-o', 'LineWidth', 1.5);
xlabel('Nombre de neurones H2 (AE2)');
ylabel('Nombre total de paramètres');
title(sprintf('AE1(H1=%d)+AE2+Softmax avec FT - Paramètres vs H2', H1));
grid on;
saveas(gcf, fullfile(resultsDir, 'opt_nParams_vs_H2.png'));

%% 4. Résumé global

fprintf('\n================= RÉSUMÉ OPTIMISATION H2 (H1 = %d) =================\n', H1);
fprintf('H2   | Acc (%%) | Err (%%) | Nb paramètres\n');
fprintf('------------------------------------------\n');
for k = 1:nConfigs
    fprintf('%4d | %7.2f | %7.2f | %10d\n', H2_list(k), acc_all(k)*100, err_all(k)*100, nParams_all(k));
end
fprintf('------------------------------------------\n');

[bestAcc, idxBest] = max(acc_all);
bestH2 = H2_list(idxBest);
bestParams = nParams_all(idxBest);

fprintf('Meilleure configuration (sur H2_list) : H1 = %d, H2 = %d\n', H1, bestH2);
fprintf(' -> Accuracy = %.2f %% | Erreur = %.2f %% | Nb paramètres = %d\n', ...
    bestAcc*100, (1-bestAcc)*100, bestParams);

fprintf('\nNB : Matrices de confusion et courbes enregistrées dans : %s\n', resultsDir);
fprintf('====================================================================\n');

%% === FONCTIONS LOCALES ===

function X = vectorizeImages(imgCell)
% vectorizeImages : transforme une cellule d'images 28x28 en matrice 784 x N
    numImg = numel(imgCell);
    [h, w] = size(imgCell{1});
    X = zeros(h*w, numImg);
    for k = 1:numImg
        img = imgCell{k};
        X(:,k) = img(:);  % vectorisation en colonne
    end
end

function [acc, err, nParams, classAcc] = evaluateNetwork(net, xTest, tTest, figPrefix, resultsDir)
% evaluateNetwork :
%   - applique le réseau net sur xTest
%   - calcule :
%       * accuracy globale
%       * erreur de classification
%       * nb total de paramètres
%       * accuracy par classe (1..10)
%   - trace et sauvegarde la matrice de confusion

    % Sorties réseau
    y = net(xTest);

    % Classes vraies / prédites (indices 1..10)
    [~, idxTrue] = max(tTest, [], 1);
    [~, idxPred] = max(y,     [], 1);

    % Accuracy / erreur globales
    acc = mean(idxTrue == idxPred);
    err = 1 - acc;

    % Nombre de paramètres
    try
        w = getwb(net);
        nParams = numel(w);
    catch
        nParams = NaN;
    end

    % Accuracy par classe
    nClasses = size(tTest,1);
    classAcc = zeros(1, nClasses);
    for c = 1:nClasses
        idxC = (idxTrue == c);
        if any(idxC)
            classAcc(c) = mean(idxPred(idxC) == c);
        else
            classAcc(c) = NaN;
        end
    end

    % Matrice de confusion
    fConf = figure('Visible','off');
    plotconfusion(tTest, y);
    title(strrep(figPrefix, '_', '\_'));
    saveas(fConf, fullfile(resultsDir, [figPrefix '_confusion.png']));
    close(fConf);
end
