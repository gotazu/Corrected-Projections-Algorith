function theta_prime=CPA(R,initial_P_diagonal,  B_basis)
% Iterative implementation of corrected projections algorithm

% ============================== INPUTS ============================== 
%   B_basis           : N X M dictionary matrix
%
%   R                 : L X N measurement matrix
%
%   initial_P_diagonal: Initialization scalar value for the P(t) diagonal matrix.
%   it is inversely proportional to the regularization constant. A value of
%   2.5 was used for simulations in [2].
%                    See Ref [2] for details.
% ============================== OUTPUTS ============================== 
%   theta_prime           : M X L presence parameter matrix. The Lth is the final presence parameters. 
%
% Created by Gonzalo Otazu, 2017
% References
%[1] Otazu GH, Leibold C. “A corticothalamic circuit model for sound identification in complex scenes.”  PLoS One. 2011;6(9):e24270.
%[2] Otazu GH. “Robust method for finding sparse solutions to linear inverse problems using an L2 regularization.” Preprint at https://arxiv.org/abs/1701.00573 (2017).

[dimension,basis_size]=size(B_basis);
[n,dimension]=size(R);

vector_size=dimension;
stimulus_length=n;
B=R';
initial_theta=zeros(basis_size,1);
initial_P_value=eye(basis_size)*initial_P_diagonal;   %big gain

%Initialize the responses
prev_theta=initial_theta;
prev_P=initial_P_value;
identity_kalman=eye(vector_size);
identity_cov=eye(basis_size);

% Just to avoid growing matrizes
theta_prime=zeros(basis_size,stimulus_length);
y_est=zeros(vector_size,stimulus_length);
for k=1:stimulus_length
    regressor_values=(B_basis')*B(:,k);
    % I will convert the regressor as a vector
    regressor_diag=diag(regressor_values,0);
    regressor=B_basis*regressor_diag;
    
    y_est(:,k)=(regressor*prev_theta);
    
    P_prime=prev_P-prev_P*regressor'*inv(identity_kalman+regressor*prev_P*regressor')*regressor*prev_P;
    
    kalman_gain=P_prime*regressor';
    
    theta_prime(:,k)=prev_theta+kalman_gain*( (B(:,k))-y_est(:,k)  );
    
    prev_theta=theta_prime(:,k);
    
    prev_P=P_prime;
end