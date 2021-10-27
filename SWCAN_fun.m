% --¡¾Function for SWCAN¡¿--

% - Title: Self-Weighted Clustering With Adaptive Neighbors
% - Journal: IEEE Transactions on Neural Networks and Learning Systems, 2020
% - Author: Feiping Nie, Danyang Wu, et.al.
% - Contact: If you have any questions about this work, please feel free to contact 
%           danyangwu41x@mail.nwpu.edu.cn, we will response to you as soon as possible.
% -If this code is helpful to you, please kindly cite our paper:
%  "Self-Weighted Clustering With Adaptive Neighbors"    
% -------------------------------------------------------------------------
function [knum, summ, W, y, A, F] = SWCAN_fun(X, c, k, r)
%--¡¾Input¡¿--
% X: dim*num data matrix, each column is a data point
% c: number of clusters
% k: number of neighbors to determine the initial graph, and the parameter r if r<=0
% r: paremeter > 0. If r<0, then it is determined by algorithm with k
%--¡¾Output¡¿--
% knum  flag: find c components?
% summ  objective value
% W:    Theta: weighted matrix
% y: num*1 cluster indicator vector
% A: num*num learned symmetric similarity matrix
% evs: eigenvalues of learned graph Laplacian in the iterations

knum = 0; 
NITER = 30;
[dim,num] = size(X); 
summ = zeros(NITER,1); 

if nargin < 4
    r = -1;
end;

%==¡¾Initialization¡¿==

%--¡¾Initialize A¡¿
distX = L2_distance_1(X,X);
[distX1, idx] = sort(distX,2);
A = zeros(num);
rr = zeros(num,1);
for i = 1:num
    di = distX1(i,2:k+2);
    rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));
    id = idx(i,2:k+2);
    A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end;

if r <= 0
    r = mean(rr);
end;
lambda = r;

%--¡¾Initialize F¡¿
A0 = (A+A')/2;
D0 = diag(sum(A0));
L0 = D0 - A0;
[F, temp, evs]=eig1(L0, num, 0);   
F = F(:,2:(c+1));
F = F./repmat(sqrt(sum(F.^2,2)),1,c); 
% F = F(:,1:(c));

%--¡¾Initialize W¡¿ 
W =  1/dim*eye(dim);


%==¡¾Iterative Update¡¿==

for iter = 1:NITER
    distf = L2_distance_1(F',F');
    distx = L2_distance_1(W'*X,W'*X);
    
%--¡¾Update A¡¿  
    A = zeros(num);
    for i=1:num
        idxa0 = 1:num;
        dfi = distf(i,idxa0);
        dxi = distx(i,idxa0);
        ad = -(dxi+lambda*dfi)/(2*r);
        A(i,idxa0) = EProjSimplex_new(ad);
    end;                         
    AA = (A+A')/2;
    D = diag(sum(AA));
    L = D-AA;
 
%--¡¾Update W¡¿
    T = X*L*X';
    temp1 = 0;
    for i = 1 : dim
    temp1 = temp1 + 1/(T(i,i));
    end
    for i = 1 : dim
    W(i,i) = 1/(T(i,i) * temp1); 
    end   
    
%--¡¾Update F¡¿
    F_old = F;  
   [F, temp, ev]=eig1(L, c, 0);
    evs(:,iter+1) = ev;
  
%--¡¾Calculate objective value¡¿
%     sum1 = 2*trace(W'*T*W);
%     sum2 = r*(norm(A,'fro'))^2;
%     sum3 = 2*lambda*trace(F'*L*F);
%     summ(iter) = sum1 + sum2 + sum3;    
   
    
%¡¾Acceleration, affect convergence¡¿    
    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1));
    if fn1 > 0.00000000001
        lambda = 2*lambda;
    elseif fn2 < 0.00000000001
        lambda = lambda/2;  F = F_old;
    else
        break
    end

end
%[labv, tem, y] = unique(round(0.1*round(1000*F)),'rows');
[clusternum, y]=graphconncomp(sparse(AA)); y = y';
if clusternum == c
    knum = 1;
else
       disp('cannot find $c$ components');
end

end


