clear all
clc


%% Problem 1
n_vertices = 9;
E = [0, 2; 
     1, 2;
     2, 3; 
     2, 4;
     2, 6;
     4, 5; 
     4, 6;
     4, 7;
     5, 6;
     6, 7;
     6, 8];

L1 = getLaplacian(E, n_vertices);
[V, D] = eig(L1);
syms t
x_0 = [10; 20; 12; 5; 30; 12; 15; 16; 25];
x_t = V(:, 1)' * x_0 * V(:, 1);

for i = 2 : 9
    x_t  = x_t + exp(-D(i, i)*t) * V(:, i)' * x_0 * V(:, i);
end