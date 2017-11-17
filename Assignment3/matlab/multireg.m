% *******************************************************************************
% Function: multireg(Theta, W, B0, A0, nu0, V0, B, Sigma)
%
% Description: Multireg update on B and Sigma
% Thetabar=W*B
%
% Y = XB+E
% Y(n*m) X(n*k) B(k*m)
%
% priors: 
%       vec(B)|Sigma ~ N(vec(Btilde, Sigma (x) A^-1);
%       Sigma ~ IW(nu, V);
%
%
% input:
%        B: kW(#demo)*npars(#theta)
%        B0: (k*m,1) vector
%        A0: (k*m,k*m) matrix
%        nu0
%        V0: m*m matrix
% Based on Rossi book, p64
% *******************************************************************************

function [B, Sigma] = multireg(Y, X, B0, A0, nu0, V0)

% dimensions
n = size(Y,1);
m = size(Y,2);
k = size(X,2);

% update values
nu = nu0 + n;

RA = chol(A0);
W = [X; RA];
Z = [Y; RA*B0];

IR = inv(chol(W'*W));
Btilde = (IR*IR') * (W'*Z);
E = Z - W*Btilde;
S = E' * E;

V = V0 + S; 

% draw Sigma now
Sigma = iwishrnd(V, nu);
CI = chol(Sigma,'lower');

% draw B conditional on Sigma
draw = normrnd(0, 1, k, m);
B = Btilde + IR * draw * CI';

end