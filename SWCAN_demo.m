% --��Demo for SWCAN��--

% - Title: Self-Weighted Clustering With Adaptive Neighbors
% - Journal: IEEE Transactions on Neural Networks and Learning Systems, 2020
% - Author: Feiping Nie, Danyang Wu, et.al.
% - Contact: If you have any questions about this work, please feel free to contact 
%           danyangwu41x@mail.nwpu.edu.cn, we will response to you as soon as possible.
% -If this code is helpful to you, please kindly cite our paper:
%  "Self-Weighted Clustering With Adaptive Neighbors"    
% -------------------------------------------------------------------------
  clc;
  clear;
  close all;

%��Load data��
  addpath(genpath('funs'))
  load('glass_uni');
  X = double(X);

%��Standardized��
  X = (prestd(X'))';         % necessary 

%��Definition��
  [~, dim] = size(X);
  c = max(Y);
  k = 12;  % average neighbors evaluation parameter 
  flag = 0; % c connected components?

%��Run��
  [knum, Summ, WW, la, A, evs] = SWCAN_fun(X', c, k);

%��Check components and evaluation��
  if knum == 1
      result = ClusteringMeasure(Y,la);
  else
      disp('cannot find $c$ components');
  end




