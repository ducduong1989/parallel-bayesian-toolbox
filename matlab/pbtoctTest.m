%% Cleanup
clc;
clf;


%% Setup
%workdir = uigetdir()
%cd(workDir);

stateDimension = 2;
obsDimension = 2;

nParticles = 500;
nObservations = 1001; % Number of observations
savedSteps = 10
startSaving = nObservations - savedSteps;


%% Initialize particle filter

weights = ones(1,nParticles) ./ nParticles;
samples = randn(stateDimension,nParticles) .* 0.01;
threshold = 0.3; %* nParticles;

a = pbtoct('initialize');
a = pbtoct('setParticles',samples, weights);
a = pbtoct('setThresholdByFactor',threshold);


%% Filtering

meas = randn(obsDimension,1).*0.01;

%results = struct('predParticles', [], ...
%    'updParticles', [],...
%    'meas', [], ...
%    'predEstimation', [], ...
%    'updEstimation', []);

measurementsX = zeros(1, nObservations);
measurementsY = zeros(1, nObservations);

estimationX = zeros(1, nObservations);
estimationY = zeros(1, nObservations);

particlesX = zeros(1, nObservations);
particlesY = zeros(1, nObservations);

for i = 1:nObservations

    % Preditcion
    a = pbtoct('predict');

    % Save intermediate results
    results(i).predEstimation = a.estimation;
    a = pbtoct('getParticles');
    results(i).predParticles = a.particles;
    
    if i >startSaving
    	for j=1:nParticles
    		particlesX(j) = a.particles(1,j);
		particlesY(j) = a.particles(2,j);
    	end
   	plot (particlesX,particlesY,".", "markersize", 4, "color", "red")
   	hold all
    end

    % Update
    meas = meas + 0.5 + randn(obsDimension,1).*0.01 + randn(obsDimension,1).*0.001;
    a = pbtoct('update',meas);
    
    % Save intermediate results
    results(i).updEstimation = a.estimation;
    estimationX(i) = a.estimation(1);
    estimationY(i) = a.estimation(2);
    
    a = pbtoct('getParticles');
    results(i).updParticles = a.particles;
    results(i).meas = meas;
    
    measurementsX(i) = meas(1);
    measurementsY(i) = meas(2);
    
    if i >startSaving
    	for j=1:nParticles
    		particlesX(j) = a.particles(1,j);
		particlesY(j) = a.particles(2,j);
    	end
   	plot (particlesX,particlesY,".", "markersize", 4, "color", "blue")
   	hold all
    end
        
end%for

title "Linear Movement"
xlabel "x in m"
ylabel "y in m"

plot (measurementsX(end-savedSteps:end),measurementsY(end-savedSteps:end),"*", "markersize", 4, "color", "green")
hold all
plot (estimationX(end-savedSteps:end),estimationY(end-savedSteps:end),"-", "markersize", 4, "color", "black")

