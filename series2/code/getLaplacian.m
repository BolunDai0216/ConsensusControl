function L = getLaplacian(E, n)
    L = zeros(n, n);
    for i = 1 : size(E, 1)
        L(E(i, 2)+1, E(i, 2)+1) = L(E(i, 2)+1, E(i, 2)+1) + 1;
        L(E(i, 1)+1, E(i, 1)+1) = L(E(i, 1)+1, E(i, 1)+1) + 1;
        L(E(i, 2)+1, E(i, 1)+1) = -1;
        L(E(i, 1)+1, E(i, 2)+1) = -1;
    end
end