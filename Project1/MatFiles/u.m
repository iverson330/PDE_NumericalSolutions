function z = u( X, Y, n )
% u(x,y), and n is a parameter to control the frequency
    z = sin(2*pi*n.*X)+ sin(2*pi*n.*Y)   + X.*X  ;
    % Plot:
    % [x,y]=meshgrid(0:1/100:1);
    % z = u(x,y,1);
    % surf(x,y,z), shading interp; colorbar
end

