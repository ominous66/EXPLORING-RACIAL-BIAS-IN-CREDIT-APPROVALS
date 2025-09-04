%% Data Preparation
hmda = xlsread('hmda.xlsx');

% deny	-- Factor. Mortgage application status. Was the mortgage denied?
% pirat	-- payments-to-income ratio
% hirat	-- inhouse expense-to-total-income ratio
% lvrat	-- loan-to-value ratio
% chist	-- Factor. consumer credit score
% mhist	-- Factor. mortgage credit score
% phist	-- Factor. Public bad credit record?
% unemp	-- 1989 Massachusetts unemployment rate in applicant's industry
% selfemp	-- Factor. Is the individual self-employed?
% insurance	-- Factor. Was the individual denied mortgage insurance?
% condomin	-- Factor. Is the unit a condominium?
% afam	-- Factor. Is the individual African-American? Yes is Black Family, No is White Family
% single	-- Factor. Is the individual single?
% hschool	-- Factor. Does the individual have a high-school diploma?
% 
deny = hmda(:,1);
pirat = hmda(:,2);
hirat = hmda(:,3);
lvrat = hmda(:,4);
chist = hmda(:,5);
mhist = hmda(:,6);
phist = hmda(:,7);
unemp = hmda(:,8);
selfemp = hmda(:,9);
insurance = hmda(:,10);
condomin = hmda(:,11);
afam = hmda(:,12);
single = hmda(:,13);
hschool = hmda(:,14);
% % 
idx = strcmp(hmda.deny, "yes");
hmda.deny(idx) = {1};
idx = strcmp(hmda.deny, "no");
hmda.deny(idx) = {0};
% 
for col = 1:width(hmda)
      if iscell(hmda{:, col}) || isstring(hmda{:, col})
          % Replace "yes" with 1
          idx_yes = strcmp(hmda{:, col}, "yes");
          hmda{idx_yes, col} = {1};
 
          % Replace "no" with 0
          idx_no = strcmp(hmda{:, col}, "no");
          hmda{idx_no, col} = {0};
      end
  end
 % 
% 
 %% Linear Regression
 y = deny;
 X = [ones(size(y)) pirat hirat lvrat chist mhist phist unemp selfemp insurance condomin afam single hschool];
 
 bols = regress(y,X)
 
 rowLabel = char('pirat', 'hirat' ,'lvrat', 'chist', 'mhist', 'phist', 'unemp' ,'selfemp', 'insurance', 'condomin', 'afam', 'single', 'hschool');
 k = size(bols , 1);
% 
 fprintf('Linear Regression Results\n\n')
 fprintf('Variable   Coefficient\n');
 fprintf('-------------------------\n')
 for i = 1:k-1
     fprintf(' %s %1.5f \n', rowLabel(i,:), bols(i));
 end
 
 %% Binary Probit Model
 burn = 2500;
 mcmc = 10000;
 R = burn + mcmc;
 k = size(X, 2);
 
 b0 = zeros(k,1);
 B0 = 100*eye(k);
 invB0 = B0\eye(k);
 invB0b0 = B0\b0;
 XX = X'*X;
 z = y;
% 
 betamat = zeros(k,mcmc);
 betamat(:,1) = bols;
% 
 tic 
 h = waitbar(0, 'Simulation In Progress');
% 
 for i = 2:R;
     betamat(:,i) = drawbeta_binary_probit(XX,invB0, z, invB0b0, X);
     z = drawz_binary_probit(z, X, betamat(:,i),y);
     waitbar(i/R);
 end
% 
 close(h)
 toc
% 
 betaPostMean = mean(betamat(:,burn+1:R),2);
 betaPostStd = std(betamat(:,burn+1:R),0,2);
% 
%% Trace plot for each coefficient
% figure;
 for j = 1:k
     subplot(ceil(k/3), 3, j);
     plot(betamat(j, burn+1:R));
     title(['Trace Plot for Beta ', num2str(j)]);
     xlabel('Iteration');
     ylabel(['Beta ', num2str(j)]);
 end
% 
% Autocorrelation plot for one coefficient (e.g., first coefficient)
 figure;
 autocorr(betamat(1, burn+1:R), 'NumLags', 50);
 title('Autocorrelation for Beta 1');
% 
 %% MLE Estimates of coefficients
 beta0 = zeros(k,1);
 [n, k] = size(X);
% 
 % Optimization options
 options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'iter');
% 
 % Maximize log-likelihood
 [beta_mle, fval, exitflag, output, grad, hessian] = fminunc(@(b) probit_loglik(b, X, y), beta0, options);
% 
 disp('Estimated Coefficients (MLE):');
 disp(beta_mle);
% 
 covMatrix = inv(hessian);
 stdErrorMLE = sqrt(diag(covMatrix));
% 
 %% Table Printing
 fprintf('Beta Estimation Results\n\n')
 fprintf('Variable   Linear Regression  MCMC Mean  MCMC Std Error  MLE Mean  MLE Std Error\n');
 fprintf('----------------------------------------------------------------------------------\n')
 for i = 1:k-1
     fprintf(' %s    %1.4f         %1.4f       %1.4f        %1.4f        %1.4f \n', rowLabel(i,:), bols(i), betaPostMean(i), betaPostStd(i), beta_mle(i), stdErrorMLE(i));
 end
% 
 %% Average Covariate Effect (ACE)
afam_ml_est = 0.3957;
% 
% % Define function to calculate Average Covariate Effect (ACE)
calculate_ace = @(beta) normpdf(0) * beta;
% 
% Calculate ACE for ML estimate
 ace_ml = calculate_ace(afam_ml_est);
% 
% Display results with 4 decimal places
fprintf('Average Covariate Effect (ACE) for ML Estimate: %.4f\n', ace_ml);

%% Classical estimation of Probit Model

% Read the table
data = readtable('hmda.xlsx', 'Sheet', 'data');

% variables
varNames = data.Properties.VariableNames;
covariateNames = setdiff(varNames, {'deny'});
X = table2array(data(:, covariateNames));
y = data.deny;

X_design = [ones(size(X,1),1) X];
[n,k] = size(X_design);

% Run the Probit Model
[b, ~, stats] = glmfit(X, y, 'binomial', 'link', 'probit');

% Store classical estimates in b_classical so we can reference later
b_classical = b;

% Display the maximum likelihood estimates and standard errors
fprintf('Classical Probit Model Estimation Results:\n');
fprintf('-------------------------------------------\n');
fprintf('%-20s %-12s %-12s\n', 'Parameter', 'Estimate', 'Std. Error');
fprintf('%-20s %-12.4f %-12.4f\n', 'Intercept', b(1), stats.se(1));
for i = 2:length(b)
    fprintf('%-20s %-12.4f %-12.4f\n', covariateNames{i-1}, b(i), stats.se(i));
end

%% Bayesian binary probit model

% MCMC settings
nIter = 12500;
burn = 2500;
nStore = 10000;
beta_store = zeros(nStore, k);

% Prior: beta ~ N(0, 100*I)
beta0 = zeros(k,1);
V0 = 100*eye(k);
V0_inv = (1/100)*eye(k);

% Initialize beta and latent variable z
beta = zeros(k,1);
z = zeros(n,1);

% MCMC sampling
for iter = 1:nIter
    
    % (1) Sample latent variables z_i
    for i = 1:n
        mu_i = X_design(i,:) * beta;
        if y(i) == 1
            a = 0;      % Truncate from 0 to +∞ if y=1
            bnd = Inf;
        else
            a = -Inf;   % Truncate from -∞ to 0 if y=0
            bnd = 0;
        end
        
        z(i) = truncated_normal_sample(mu_i, 1, a, bnd);
    end
    
    % Sample beta from its full conditional
    % beta ~ N(m_post, V_post)
    V_post = inv(X_design' * X_design + V0_inv);
    m_post = V_post * (X_design' * z + V0_inv * beta0);
    beta = m_post + chol(V_post, 'lower') * randn(k,1);
    
    % Store draws after burn-in
    if iter > burn
        beta_store(iter - burn, :) = beta';
    end
end

% Compute posterior summary statistics
posterior_mean = mean(beta_store)';
posterior_std = std(beta_store)';

%% 4. Display Combined Results

fprintf('\n\nCombined Results: ML Estimates and Bayesian Posterior Summaries\n');
fprintf('-----------------------------------------------------------------------\n');
fprintf('%-15s %-12s %-12s %-12s %-12s\n', 'Parameter', 'ML_Est', 'ML_SE', 'Bayes_Mean', 'Bayes_SD');
fprintf('%-15s %-12.4f %-12.4f %-12.4f %-12.4f\n', ...
    'Intercept', b_classical(1), stats.se(1), posterior_mean(1), posterior_std(1));

for j = 2:k
    fprintf('%-15s %-12.4f %-12.4f %-12.4f %-12.4f\n', ...
        covariateNames{j-1}, b_classical(j), stats.se(j), ...
        posterior_mean(j), posterior_std(j));
end

%% 5. Trace Plot of MCMC Draws for All Parameters
figure;
nRows = ceil(sqrt(k));
nCols = ceil(k / nRows);
for j = 1:k
    subplot(nRows, nCols, j);
    plot(beta_store(:,j));
    title(sprintf('Param %d', j));
    xlabel('Iteration');
    ylabel('Value');
end
sgtitle('Trace Plots for MCMC Draws of Probit Model Parameters');


%% --- Helper Function: Truncated Normal Sampler ---
function x = truncated_normal_sample(mu, sigma, a, b)
% Samples one draw from a N(mu, sigma^2) distribution truncated to [a, b].
% Simple rejection sampler for moderate truncation.
    accepted = false;
    while ~accepted
        x_candidate = mu + sigma*randn;
        if (x_candidate >= a) && (x_candidate <= b)
            x = x_candidate;
            accepted = true;
        end
    end
end
