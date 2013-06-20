%% Cleanup
clc;

%% Setup
workDir = '..\bin\Debug\';
%workDir = uigetdir()
cd(workDir);

nParticles = 500;
stateDimension = 9;
obsDimension = 9;
nObservations = 1001; % Number of observations

%% Initialize particle filter

weights = ones(1,nParticles) ./ nParticles;
samples = randn(stateDimension,nParticles) .* 0.01;
threshold = 0.3; %* nParticles;

a = pbtmex('initialize');
a = pbtmex('setParticles', samples, weights);
a = pbtmex('setThresholdByFactor', threshold);


%% Filtering

meas = randn(obsDimension,1).*0.01;

measurementsX = zeros(1, nObservations);
measurementsY = zeros(1, nObservations);

estimationX = zeros(1, nObservations);
estimationY = zeros(1, nObservations);

particlesX = zeros(1, nObservations);
particlesY = zeros(1, nObservations);

for i = 1:nObservations

    % Preditcion
    a = pbtmex('predict');

    % Save intermediate results
    results(i).predEstimation = a.estimation;
    a = pbtmex('getParticles');
    results(i).predParticles = a.particles;

    % Update
    meas = meas + 0.5 + randn(obsDimension,1).*0.01 + randn(obsDimension,1).*0.001;
    a = pbtmex('update',meas);
    
    % Save intermediate results
    results(i).updEstimation = a.estimation;
    estimationX(i) = a.estimation(1);
    estimationY(i) = a.estimation(2);
    
    a = pbtmex('getParticles');
    results(i).updParticles = a.particles;
    results(i).meas = meas;
    
    measurementsX(i) = meas(1);
    measurementsY(i) = meas(2);
        
end%for

plot (measurementsX,measurementsY)
hold all
plot (estimationX,estimationY)

