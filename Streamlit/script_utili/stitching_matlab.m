% Pulizia del workspace e delle variabili
clear
clc

% Caricamento delle immagini
buildingDir = fullfile('Inserire il path della cartella contenente i frame per lo stitching');
buildingScene = imageDatastore(buildingDir);

% Stampa delle immagini che verranno usate per lo stitching
%montage(buildingScene.Files)

% Lettura della prima immagine dell'image set
I = readimage(buildingScene,1);

% Inizializzazione delle features
grayImage = im2gray(I);
points = detectSURFFeatures(grayImage);
[features, points] = extractFeatures(grayImage,points);

% Inizializza tutte le trasformazioni alla matrice identità
numImages = numel(buildingScene.Files);
tforms(numImages) = projective2d(eye(3));

% Inizializzazione delle variabili per mantenere le dimensioni
% dell'immagine
imageSize = zeros(numImages,2);

% Iterazione, a coppie di due, sulle immagini restanti
for n = 2:numImages
    
    % Memorizzazione dei punti e delle features
    pointsPrevious = points;
    featuresPrevious = features;
        
    % Lettura dell'immagine n
    I = readimage(buildingScene, n);
    
    % Conversione dell'immagine in scala di grigi
    grayImage = im2gray(I);    
    
    % Salvataggio delle misure dell'immagine
    imageSize(n,:) = size(grayImage);
    
    % Riconoscimento ed estrazione delle features per applicare SURF all'immagine n
    points = detectSURFFeatures(grayImage);    
    [features, points] = extractFeatures(grayImage, points);
  
    % Ricerca delle corrispondenze tra l'immagine n e l'immagine n-1
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);
       
    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);        
    
    % Stima della trasformazione tra l'immagine n e l'immagine n-1
    tforms(n) = estimateGeometricTransform2D(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
    
    % Calcolo T(n) * T(n-1) * ... * T(1)
    tforms(n).T = tforms(n).T * tforms(n-1).T; 
end

% Calcolo dei bordi per ogni trasformazione in output
for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);    
end

avgXLim = mean(xlim, 2);
[~,idx] = sort(avgXLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx);

Tinv = invert(tforms(centerImageIdx));
for i = 1:numel(tforms)    
    tforms(i).T = tforms(i).T * Tinv.T;
end

for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

% Ricerca del minimo e del massimo inerente ai bordi dell'output 
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Larghezza e altezza dell'immagine panoramica ottenuta
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Inizializzazione dell'immagine panoramica "vuota"
panorama = zeros([height width 3], 'like', I);

blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');  

% Creazione di un oggetto 2D con le misure prima definite
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Creazione dell'immagine panoramica
for i = 1:numImages
    
    I = readimage(buildingScene, i);   
   
    % Trasformazione dell'immagine i nell'immagine panoramica
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
                  
    % Creazione di una maschera binaria    
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
    
    % Sovrapposizione dell'immagine elaborata al panorama
    panorama = step(blender, panorama, warpedImage, mask);
end

figure('Name','Immagine paronamica ottenuta','NumberTitle','off', 'Color','black');
imshow(panorama)