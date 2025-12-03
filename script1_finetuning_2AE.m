%% TD6 - Étude de l'effet du fine-tuning et de l'ajout d'un deuxième autoencodeur
% On teste 4 cas avec la même architecture de base :
% Cas 1 : autoencoder1 + softmax (sans fine-tuning)
% Cas 2 : autoencoder1 + softmax (avec  fine-tuning)
% Cas 3 : autoencoder1 + autoencoder2 + softmax (sans fine-tuning)
% Cas 4 : autoencoder1 + autoencoder2 + softmax (avec  fine-tuning)
%
% Objectif :
% - comparer Cas1 vs Cas2 : effet du fine-tuning sur une architecture à 1 AE
% - comparer Cas3 vs Cas4 : effet du fine-tuning sur une architecture à 2 AE
% - comparer Cas2 vs Cas4 : effet de l'ajout d'un 2ème AE (à architecture équivalente)

clear; close all; clc;

%% 0. Paramètres généraux

rng('default');  % pour la reproductibilité

resultsDir = 'results_ft_2AE';
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

% Taille des couches cachées (même architecture pour les 4 cas)
hiddenSize1 = 100;  % nombre de neurones de l'encodeur 1
hiddenSize2 = 50;   % nombre de neurones de l'encodeur 2

% Hyperparamètres des autoencodeurs (à adapter si besoin)
maxEpochsAE   = 100;
L2reg         = 0.001;
sparsityReg   = 4;
sparsityProp  = 0.15;

fprintf('================= PARAMÈTRES GÉNÉRAUX =================\n');
fprintf('hiddenSize1 (AE1) = %d neurones\n', hiddenSize1);
fprintf('hiddenSize2 (AE2) = %d neurones\n', hiddenSize2);
fprintf('MaxEpochs (AE)    = %d\n', maxEpochsAE);
fprintf('L2                = %.4f\n', L2reg);
fprintf('SparsityReg       = %.2f\n', sparsityReg);
fprintf('SparsityProp      = %.2f\n', sparsityProp);
fprintf('========================================================\n\n');

%% 1. Chargement et préparation des données

[xTrainImages, tTrain] = digitTrainCellArrayData;
[xTestImages,  tTest]  = digitTestCellArrayData;

numTrain = numel(xTrainImages);
numTest  = numel(xTestImages);
fprintf('Nombre d''images d''apprentissage : %d\n', numTrain);
fprintf('Nombre d''images de test          : %d\n\n', numTest);

% Vectorisation des images pour les réseaux empilés
xTrain = vectorizeImages(xTrainImages);  % 784 x numTrain
xTest  = vectorizeImages(xTestImages);   % 784 x numTest

%% 2. Entraînement de l'autoencodeur 1 (AE1)

fprintf('=== Entraînement de l''autoencodeur 1 (AE1) ===\n');
autoenc1 = trainAutoencoder( ...
    xTrainImages, ...
    hiddenSize1, ...
    'MaxEpochs',             maxEpochsAE, ...
    'L2WeightRegularization', L2reg, ...
    'SparsityRegularization', sparsityReg, ...
    'SparsityProportion',     sparsityProp, ...
    'ScaleData',              false);

% Extraction des caractéristiques de niveau 1
feat1Train = encode(autoenc1, xTrainImages);

%% 3. Cas 1 : AE1 + Softmax (sans fine-tuning)

fprintf('\n=== Cas 1 : AE1 + Softmax (sans fine-tuning) ===\n');
fprintf('Architecture : AE1(hiddenSize1 = %d) + Softmax\n', hiddenSize1);

% Entraînement de la couche softmax sur les features de AE1
softnet1 = trainSoftmaxLayer(feat1Train, tTrain, 'MaxEpochs', 100);

% Réseau empilé Cas 1 (AE1 + softmax)
net_c1 = stack(autoenc1, softnet1);

% Évaluation sur la base de test (sans fine-tuning)
[acc_c1, err_c1, nParams_c1, classAcc_c1] = evaluateNetwork( ...
    net_c1, xTest, tTest, ...
    sprintf('cas1_AE1_softmax_noFT_H1_%d', hiddenSize1), ...
    resultsDir);

fprintf('Cas 1 - Accuracy globale = %.2f %% | Erreur = %.2f %% | Nb paramètres = %d\n', ...
    acc_c1*100, err_c1*100, nParams_c1);
fprintf('Cas 1 - Accuracy par classe (1..10) = [');
fprintf(' %.2f', classAcc_c1*100);
fprintf(' ] %%\n');

%% 4. Cas 2 : AE1 + Softmax (avec fine-tuning)

fprintf('\n=== Cas 2 : AE1 + Softmax (avec fine-tuning) ===\n');
fprintf('Architecture : AE1(hiddenSize1 = %d) + Softmax (fine-tuning global)\n', hiddenSize1);

% On part du réseau Cas 1 et on fait un fine-tuning global
net_c2 = train(net_c1, xTrain, tTrain);

[acc_c2, err_c2, nParams_c2, classAcc_c2] = evaluateNetwork( ...
    net_c2, xTest, tTest, ...
    sprintf('cas2_AE1_softmax_FT_H1_%d', hiddenSize1), ...
    resultsDir);

fprintf('Cas 2 - Accuracy globale = %.2f %% | Erreur = %.2f %% | Nb paramètres = %d\n', ...
    acc_c2*100, err_c2*100, nParams_c2);
fprintf('Cas 2 - Accuracy par classe (1..10) = [');
fprintf(' %.2f', classAcc_c2*100);
fprintf(' ] %%\n');

%% 5. Entraînement de l'autoencodeur 2 (AE2) sur les features de AE1

fprintf('\n=== Entraînement de l''autoencodeur 2 (AE2) ===\n');
fprintf('Architecture AE2 : entrée = %d (features de AE1), hiddenSize2 = %d\n', ...
    hiddenSize1, hiddenSize2);

autoenc2 = trainAutoencoder( ...
    feat1Train, ...
    hiddenSize2, ...
    'MaxEpochs',             maxEpochsAE, ...
    'L2WeightRegularization', L2reg, ...
    'SparsityRegularization', sparsityReg, ...
    'SparsityProportion',     sparsityProp, ...
    'ScaleData',              false);

% Extraction des caractéristiques de niveau 2
feat2Train = encode(autoenc2, feat1Train);

%% 6. Cas 3 : AE1 + AE2 + Softmax (sans fine-tuning)

fprintf('\n=== Cas 3 : AE1 + AE2 + Softmax (sans fine-tuning) ===\n');
fprintf('Architecture : AE1(H1 = %d) + AE2(H2 = %d) + Softmax\n', hiddenSize1, hiddenSize2);

% Entraînement de la softmax sur les features de AE2
softnet2 = trainSoftmaxLayer(feat2Train, tTrain, 'MaxEpochs', 100);

% Réseau empilé Cas 3 (AE1 + AE2 + softmax)
net_c3 = stack(autoenc1, autoenc2, softnet2);

[acc_c3, err_c3, nParams_c3, classAcc_c3] = evaluateNetwork( ...
    net_c3, xTest, tTest, ...
    sprintf('cas3_AE1_AE2_softmax_noFT_H1_%d_H2_%d', hiddenSize1, hiddenSize2), ...
    resultsDir);

fprintf('Cas 3 - Accuracy globale = %.2f %% | Erreur = %.2f %% | Nb paramètres = %d\n', ...
    acc_c3*100, err_c3*100, nParams_c3);
fprintf('Cas 3 - Accuracy par classe (1..10) = [');
fprintf(' %.2f', classAcc_c3*100);
fprintf(' ] %%\n');

%% 7. Cas 4 : AE1 + AE2 + Softmax (avec fine-tuning)

fprintf('\n=== Cas 4 : AE1 + AE2 + Softmax (avec fine-tuning) ===\n');
fprintf('Architecture : AE1(H1 = %d) + AE2(H2 = %d) + Softmax (fine-tuning global)\n', ...
    hiddenSize1, hiddenSize2);

% Fine-tuning global du réseau Cas 3
net_c4 = train(net_c3, xTrain, tTrain);

[acc_c4, err_c4, nParams_c4, classAcc_c4] = evaluateNetwork( ...
    net_c4, xTest, tTest, ...
    sprintf('cas4_AE1_AE2_softmax_FT_H1_%d_H2_%d', hiddenSize1, hiddenSize2), ...
    resultsDir);

fprintf('Cas 4 - Accuracy globale = %.2f %% | Erreur = %.2f %% | Nb paramètres = %d\n', ...
    acc_c4*100, err_c4*100, nParams_c4);
fprintf('Cas 4 - Accuracy par classe (1..10) = [');
fprintf(' %.2f', classAcc_c4*100);
fprintf(' ] %%\n');

%% 8. Résumé des résultats (effet FT + effet 2ème AE)

fprintf('\n================= RÉSUMÉ GLOBAL =================\n');
fprintf('Architecture                    | FT  | Acc (%%) | Err (%%) | Nb params\n');
fprintf('---------------------------------------------------------------------\n');
fprintf('Cas 1 : AE1 + Softmax           | Non | %6.2f | %6.2f | %8d\n', ...
    acc_c1*100, err_c1*100, nParams_c1);
fprintf('Cas 2 : AE1 + Softmax           | Oui | %6.2f | %6.2f | %8d\n', ...
    acc_c2*100, err_c2*100, nParams_c2);
fprintf('Cas 3 : AE1 + AE2 + Softmax     | Non | %6.2f | %6.2f | %8d\n', ...
    acc_c3*100, err_c3*100, nParams_c3);
fprintf('Cas 4 : AE1 + AE2 + Softmax     | Oui | %6.2f | %6.2f | %8d\n', ...
    acc_c4*100, err_c4*100, nParams_c4);
fprintf('---------------------------------------------------------------------\n');

fprintf('\nEffet du fine-tuning (AE1 seul) : Cas2 - Cas1 = %.2f points d''accuracy\n', ...
    (acc_c2 - acc_c1)*100);
fprintf('Effet du fine-tuning (AE1+AE2)  : Cas4 - Cas3 = %.2f points d''accuracy\n', ...
    (acc_c4 - acc_c3)*100);
fprintf('Effet d''ajouter AE2 (après FT)  : Cas4 - Cas2 = %.2f points d''accuracy\n', ...
    (acc_c4 - acc_c2)*100);

fprintf('\nNB : Les matrices de confusion et figures sont enregistrées dans : %s\n', resultsDir);
fprintf('=====================================================================\n');

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
%       * nombre total de paramètres
%       * accuracy par classe (1..10)
%   - trace et sauvegarde la matrice de confusion

    % Sorties du réseau
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
            classAcc(c) = NaN; % au cas où une classe n'apparaît pas
        end
    end

    % Matrice de confusion (figure)
    fConf = figure('Visible','off');
    plotconfusion(tTest, y);
    title(strrep(figPrefix, '_', '\_'));
    saveas(fConf, fullfile(resultsDir, [figPrefix '_confusion.png']));
    close(fConf);
end

