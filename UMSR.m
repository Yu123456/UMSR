%
%   X:          every colume is a sample
%   bNorm:      if or not need Normalization
%   knn:        number of neighbours for construct SIG
%   gnd:        ground truth
%   maxIter:    number of iterations
%   lamda:      paramerters
%
%
%
function result = UMSR(X, bNorm, knn, gnd, maxIter, lamda)    
    viewNum = numel(X);
    sampleNum = size(X{1},2);
    cluNum = length(unique(gnd));
    initial_step_size = 0.1;  % 初始步长较大
    min_step_size = 0.01;     % 最小步长
    islocal = 1; % only update the similarities of neighbors if islocal=1

    %% Normalization
    if bNorm == 1
        parfor i = 1:viewNum
            for  j = 1:sampleNum
                normItem = std(X{i}(:,j));
                if (0 == normItem)
                    normItem = eps;
                end
                X{i}(:,j) = (X{i}(:,j)-mean(X{i}(:,j)))/(normItem);
            end
        end
    end
    
    %% initialize S: Constructing the SIG matrices
    S = cell(1,viewNum);
    parfor v = 1:viewNum
        [S{v}, ~] = InitializeSIGs(X{v}, knn, 0);
    end
    S_initial = S;
    
    
%% initialize U D L(SumL)
    U = zeros(sampleNum);    % U为n*n矩阵
    for v = 1:viewNum
         U = U + S{v};
    end
    U = U/viewNum;            %初始U除以视图个数m
    for j = 1:sampleNum
         U(j,:) = U(j,:)/sum(U(j,:));  %对矩阵 U 的第 j 行进行归一化
    end

   %% initialize sU(对称矩阵) D L SumL
    sU = (U+U')/2;
    D = diag(sum(sU));
    SumL = D - sU;
    %disp(size(SumL));
   
    %% initialize w
    w = ones(1, viewNum).*(1/viewNum);
    
    %% initialize F
    [F, ~, ~]=eig1(SumL, cluNum, 0, 0);
    
    %% initialize Y
    tmp = diag(diag(F * F'));
    Y0 = tmp^0.5 * F;
    Y = zeros(sampleNum, cluNum);
    [~, I] = max(Y0, [], 2);
    
    for i = 1:sampleNum
        Y(i, I(i)) = 1;
    end
    C = bestMap(gnd,I);
    ACC = length(find(gnd == C))/length(gnd);

    fprintf('ACC = %.4f', ACC);
  
    error = zeros(maxIter,1);
   

    old_cost = 0;
    fz = 1e-6;
    bCalLost = 1;
    for iter = 1:maxIter
        tic;
     %% 动态步长调整
        step_size = max(min_step_size, initial_step_size * (1 - iter / maxIter));
        %% update S_v        
        for v = 1:viewNum
            DD = zeros(sampleNum);
            for i = 1:sampleNum
                for j = 1:sampleNum
                    DD(i,j) = norm(X{v}(:,i) - X{v}(:,j))^2;
                end
            end
            [~, idx] = sort(DD, 2); % sort each row

            S{v} = zeros(sampleNum);
            for i = 1:sampleNum
                id = idx(i, 2 : knn + 2);
                di = DD(i, id);
                % 修改
                numerator = di(knn+1)-di+2*w(v)*U(i,id(:))-2*w(v)*U(i,id(knn+1));
                denominator1 = knn*di(knn+1)-sum(di(1:knn));
                denominator2 = 2*w(v)*sum(U(i,id(1:knn)))-2*knn*w(v)*U(i,id(knn+1));
                S{v}(i,id) = max(numerator/(denominator1+denominator2+eps),0);
            end

            tc1 = sum(S{v}, 2);% here test it sum to 1
        end

        % update w
        for v = 1:viewNum
            US = U - S{v};
            distUS = norm(US, 'fro')^2;
            if distUS == 0
                distUS = eps;
            end
            w(v) = 0.5/sqrt(distUS);
        end


        % update U
        dist = L2_distance_1(F',F');
        U = zeros(sampleNum);
        for i=1:sampleNum
            idx = zeros();
            for v = 1:viewNum
                s0 = S{v}(i,:);
                idx = [idx,find(s0>0)];
            end
            idxs = unique(idx(2:end));
            if islocal == 1
                idxs0 = idxs;
            else
                idxs0 = 1:sampleNum;
            end
            for v = 1:viewNum
                s1 = S{v}(i,:);
                si = s1(idxs0);
                di = dist(i,idxs0);
                mw = viewNum*w(v);
                lmw = lamda(2)/mw;
                q(v,:) = si-0.5*lmw*di;
            end
            U(i,idxs0) = SloutionToP19(q,viewNum);
            clear q;
        end
        

        
        %% update R
        dlg = diag(Y' * Y);      
        dlg(find(dlg == 0)) = eps;
        tmp = diag(dlg);
        Dlg_Y = (tmp)^(-0.5);
     % Initialize R as an identity matrix or with small random values
        if iter == 1
           R = eye(cluNum) * 1e-2;  % Small random values for better stability
        else
    % Perform SVD to initialize R
           M = Dlg_Y * Y' * F;
           [rU,~,rV] = svd(M, 'econ');
           R = rV * rU';
    % Ensure R remains orthogonal
           [U_orth,~,V_orth] = svd(R);
           R = U_orth * V_orth';
        end
       %M = Dlg_Y *  Y' * F;
       %[rU,~,rV] = svd(M);
       %R = rV * rU';
        
        %% update F
        % update F with regularization
          SumA = SumL + lamda(2) * eye(sampleNum);  % Regularization term added
          b = (lamda(3) / lamda(2)) * Y * R;
          [F, cost] = GPI(SumA, F, b);

        
        last_F_cost = 0;
        cost = 1e-4;
        times = 0;
        
        %tic;
        while (cost - last_F_cost > 1e-4 && times < 20)
            last_F_cost = cost;
            [F, cost] = GPI(SumA, F, b);  
            times = times + 1;
        end
      
        %fprintf('GPI time:%.4f\n', toc);
        
         
        %% update Y
        Y = zeros(sampleNum, cluNum);
        G = F * R;
        N1 = zeros(cluNum, 1); %n_j
        N2 = zeros(cluNum, 1); %y_j
        parfor i = 1:cluNum
            N1(i) = Y(:, i)' * Y(:, i);
        end
        
        H = Y .* G;        
        parfor i = 1:cluNum
            tmp = sum(H,1);
            N2(i) = tmp(i);
        end
                
        for i = 1:sampleNum   
            maxValue = 0;
            maxJ = 0;
            for j = 1:cluNum
                tmp1 = N2(j) + (1 - Y(i, j)) * G(i, j);
                tmp2 = max((N1(j) + (1 - Y(i, j)))^0.5, eps);
                tmp3 = N2(j) - Y(i, j) * G(i, j);
                tmp4 = max((N1(j)  - Y(i, j))^0.5, eps);
                curValue = tmp1 / tmp2 - tmp3 / tmp4;
                if (maxValue < curValue)
                    maxValue = curValue;
                    maxJ = j;
                end
            end
            if (maxJ ~= 0)
                Y(i, maxJ) = 1;
            else
                fprintf('err!\n');
            end
        end    

         
        
         %% get result
        [~, I] = max(Y, [], 2);          

        [acc, nmi, Pu] = ClusteringMeasure(gnd,I);
        ARI = RandIndex(gnd,I); % ARI
        [Fscore,Precision,Recall] = compute_f(gnd,I);% F-score   Precision  Recall
        result = [acc, nmi, ARI, Fscore, Pu, Precision, Recall, iter];

        str = sprintf('[iter:%d] [ACC %.2f] [NMI %.2f] [ARI %.2f] [F-score %.2f] [Purity %.2f] [Precision %.2f] [Recall %.2f][lamda2:%.3f lamda3:%.3f] [Exe Time:%.2f] %s',...
            iter, acc * 100,  nmi * 100, ARI * 100, Fscore * 100, Pu * 100, Precision * 100, Recall * 100, lamda(2), lamda(3), toc, GetTimeStrForLog());
        fprintf('%s\n',str);      
    
        %% update lost        
       % 定义一个数组存储所有迭代的 cost
all_cost = zeros(maxIter, 1);

for iter = 1:maxIter
    tic;
    
    %% 在这里记录每次迭代的 cost 值
    if (bCalLost == 1)
        % 更新损失值
        if (maxIter > 1) 
            error(iter) = cost_function(X, S, F, SumL, R, Y, lamda);   
            diff = abs(old_cost - error(iter));  % 计算与上次损失的差值

            % 判断是否收敛，若差值足够小则提前终止
            if (iter > 2 && diff / error(iter) < fz)
                fprintf('diff: %.4f error(iter): %.4f old_cost: %.4f diff/error(iter): %.6f\n', ...
                    diff, error(iter), old_cost, diff / error(iter));
                break;  % 收敛，提前终止
            end
            old_cost = error(iter);  % 更新 old_cost
        else
            % 如果是第一轮迭代
            % fprintf('\n');
        end
    end
    
    % 记录当前的 cost
    all_cost(iter) = error(iter);

end
        
    end    
    clear F Y S R; 
    
end

function [F, cost] = GPI(SumA, F, b)
    %for i = 1:20
        M = 2 * SumA * F + 2 * b;
        [U,~,V] = svd(M,'econ');
        F = U * V';
        cost = F_cost(SumA, F, b);
        %fprintf('F_cost:%.6f\n', cost);
    %end

end

function err = F_cost(SumA, F, b)
    err = trace(F' * SumA * F + 2 * F' *b);
end

function err = cost_function(X, S, F, sumL, R, Y, lamda)
	sampleNum = size(X{1},2);
    sum0 = 0;
    sum1 = 0;
    viewNum = numel(X);
    parfor v = 1:viewNum
        for i = 1:sampleNum
            for j = 1:sampleNum
                nor = norm(X{v}(:, i) - X{v}(:, j));
                sum0 = sum0  + nor^2 * S{v}(i,j);
            end
        end
    end
    
    parfor v = 1:viewNum
        for i = 1:sampleNum
           sum1 = sum1 + norm(S{v}(:, i))^2;
        end
    end
    
    sum2 = trace(F' * sumL * F);
    
    dlg = diag(Y' * Y);      
    dlg(find(dlg == 0)) = eps;
    tmp = diag(dlg);
    Dlg_Y = (tmp)^(-0.5);       
    tmp = F * R - Y * Dlg_Y;
    %tmp = F * R - Y * ((Y' * Y)^(-0.5));
        
    sum3 = (norm(tmp, 'fro'))^2;
     
    err = sum0 + lamda(1) * sum1 + 2 * lamda(2) * sum2 + lamda(3) * sum3;
     str = sprintf(' [sum0 and sum1]:[%.4f] [sum2]:[%.4f]  [sum3]:[%.4f] [cost:%.4f]', sum0 + lamda(1) * sum1, lamda(2) * sum2, lamda(3) * sum3, err);
     str = sprintf('[all cost:%.4f] [sum0:%.4f] [sum1:%.4f] [sum2:%.4f] [sum3:%.4f]', err, sum0, sum1, sum2, sum3);
     fprintf('%s', str);

    
end



