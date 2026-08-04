classdef influencefunctions
    % Conditional moments, skewness, and their influence functions.
    % Port of influencefunctions.R. Call as influencefunctions.skew_infl(x).
    % All influence functions return an n-by-1 vector, except beta_infl,
    % which returns n-by-k. Estimators return scalars, except beta (k-by-1).
    % Requires R2016b or later (implicit expansion in beta_infl).

    methods (Static)

        % Conditional mean and covariance ---------------------------------

        function res = mean_cond(x, dum)
            % E(x | dum = 1), dum = {0, 1}
            x = double(x(:)); dum = double(dum(:));
            res = mean(x .* dum) / mean(dum);
        end

        function res = cov_cond(x, y, dum)
            % cov(x, y | dum = 1), dum = {0, 1}
            x = double(x(:)); y = double(y(:)); dum = double(dum(:));
            mu_dum = mean(dum);
            res = mean(x .* y .* dum) / mu_dum ...
                  - (mean(x .* dum) / mu_dum) * (mean(y .* dum) / mu_dum);
        end

        function res = beta(x, y)
            % Slope in the linear regression y = x*beta + u
            X = double(x); y = double(y(:));
            res = (X' * X) \ (X' * y);
        end

        function res = skew(x)
            % E[(x - E[x])^3] / E[(x - E[x])^2]^(3/2)
            x = double(x(:));
            d = x - mean(x);
            res = mean(d.^3) / mean(d.^2)^(3/2);
        end

        function res = skew_cond(x, dum)
            % Conditional skewness, dum = {0, 1}
            x = double(x(:)); dum = double(dum(:));
            mu_dum = mean(dum);
            mu_c_x3 = mean(x.^3 .* dum) / mu_dum;
            mu_c_x2 = mean(x.^2 .* dum) / mu_dum;
            mu_c_x  = mean(x .* dum) / mu_dum;
            F_dum = mu_c_x3 - 3*mu_c_x2*mu_c_x + 2*mu_c_x^3;
            G_dum = mu_c_x2 - mu_c_x^2;
            res = F_dum / G_dum^(3/2);
        end

        % Influence functions ---------------------------------------------

        function infl = mean_infl(x)
            % Influence function for the mean
            x = double(x(:));
            infl = x - mean(x);
        end

        function infl = beta_infl(x, y)
            % Influence function for OLS coefficients, y = x*beta + u
            X = double(x); y = double(y(:));
            n = numel(y);
            b = (X' * X) \ (X' * y);
            u_hat = y - X * b;
            % infl = (X .* u_hat) * inv(X'X / n)
            infl = (X .* u_hat) / ((X' * X) / n);
        end

        function infl = cov_infl(x, y)
            % Influence function for cov(x, y)
            x = double(x(:)); y = double(y(:));
            mu_xy = mean(x .* y); mu_x = mean(x); mu_y = mean(y);
            infl = x .* y - mu_xy - mu_y*(x - mu_x) - mu_x*(y - mu_y);
        end

        function infl = mean_cond_infl(x, dum)
            % Influence function for E(x | dum = 1), dum = {0, 1}
            x = double(x(:)); dum = double(dum(:));
            mu_dum = mean(dum);
            mu_x_dum = mean(x .* dum);
            infl = (mu_dum*(x .* dum - mu_x_dum) ...
                    - mu_x_dum*(dum - mu_dum)) / mu_dum^2;
        end

        function infl = mean_ratio_infl(x, y)
            % Influence function for E(x) / E(y)
            x = double(x(:)); y = double(y(:));
            mu_x = mean(x); mu_y = mean(y);
            infl = (mu_y*(x - mu_x) - mu_x*(y - mu_y)) / mu_y^2;
        end

        function infl = cov_cond_infl(x, y, dum)
            % Influence function for cov(x, y | dum = 1), dum = {0, 1}
            x = double(x(:)); y = double(y(:)); dum = double(dum(:));
            mu_dum = mean(dum);
            mu_x_y_dum = mean(x .* y .* dum);
            mu_x_dum = mean(x .* dum);
            mu_y_dum = mean(y .* dum);
            infl = (mu_dum*(x .* y .* dum - mu_x_y_dum) ...
                    - mu_x_y_dum*(dum - mu_dum)) / mu_dum^2 ...
                   - mu_y_dum/mu_dum * (mu_dum*(x .* dum - mu_x_dum) ...
                    - mu_x_dum*(dum - mu_dum)) / mu_dum^2 ...
                   - mu_x_dum/mu_dum * (mu_dum*(y .* dum - mu_y_dum) ...
                    - mu_y_dum*(dum - mu_dum)) / mu_dum^2;
        end

        function infl = skew_infl(x)
            % Influence function for skewness
            x = double(x(:));

            mu_x3 = mean(x.^3);
            mu_x2 = mean(x.^2);
            mu_x  = mean(x);
            F = mu_x3 - 3*mu_x2*mu_x + 2*mu_x^3;
            G = mu_x2 - mu_x^2;

            psi_x  = x - mu_x;
            psi_x2 = x.^2 - mu_x2;
            psi_x3 = x.^3 - mu_x3;
            psi_F = psi_x3 - 3*(psi_x2*mu_x + mu_x2*psi_x) + 6*mu_x^2*psi_x;
            psi_G = psi_x2 - 2*mu_x*psi_x;

            infl = (psi_F*G^(3/2) - 3/2*F*G^(1/2)*psi_G) / G^3;
        end

        function infl = skew_cond_infl(x, dum)
            % Influence function for conditional skewness, dum = {0, 1}
            x = double(x(:)); dum = double(dum(:));

            x_dum  = x .* dum;
            x2_dum = x.^2 .* dum;
            x3_dum = x.^3 .* dum;

            mu_dum = mean(dum);
            mu_x3_dum = mean(x3_dum);
            mu_x2_dum = mean(x2_dum);
            mu_x_dum  = mean(x_dum);
            mu_c_x3 = mu_x3_dum / mu_dum;
            mu_c_x2 = mu_x2_dum / mu_dum;
            mu_c_x  = mu_x_dum / mu_dum;
            F_dum = mu_c_x3 - 3*mu_c_x2*mu_c_x + 2*mu_c_x^3;
            G_dum = mu_c_x2 - mu_c_x^2;

            psi_x_dum  = ((x_dum - mu_x_dum)*mu_dum ...
                          - mu_x_dum*(dum - mu_dum)) / mu_dum^2;
            psi_x2_dum = ((x2_dum - mu_x2_dum)*mu_dum ...
                          - mu_x2_dum*(dum - mu_dum)) / mu_dum^2;
            psi_x3_dum = ((x3_dum - mu_x3_dum)*mu_dum ...
                          - mu_x3_dum*(dum - mu_dum)) / mu_dum^2;
            psi_F_dum = psi_x3_dum ...
                        - 3*(psi_x2_dum*mu_x_dum + mu_x2_dum*psi_x_dum) ...
                        + 6*mu_x_dum^2*psi_x_dum;
            psi_G_dum = psi_x2_dum - 2*mu_x_dum*psi_x_dum;

            infl = (psi_F_dum*G_dum^(3/2) ...
                    - 3/2*F_dum*G_dum^(1/2)*psi_G_dum) / G_dum^3;
        end

    end
end
