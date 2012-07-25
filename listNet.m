function listNet(dataset)
NDCGTR = zeros(5, 10); % 记录每个文件夹train的NDCG
NDCGVA = zeros(5, 10); % 记录每个文件夹vali的NDCG
NDCGTE = zeros(5, 10); % 记录每个文件夹test的NDCG
outfile = 'out.txt';   % NDCG输出的文件夹
T = 500;               % 迭代次数
times = 1;             % 每多少次记录一个模型的w
rate = 0.01;           % 步长

% 对每个文件夹进行处理
for fold = 1 : 5
    dname = [dataset '/Fold' num2str(fold) '/'];    % train的文件名
    [X, Y] = read_letor([dname '/train.txt']);      % 读入数据
    w = zeros(length(X(1, :)), 1);                  % 初始化w，全为0
    param = zeros(length(w), T / times);            % 存放每T/times次的w
    in = 1;                                         % param当前的index
    
    % 迭代T次
    for loop = 1 : T
        cnt = 1;                                    % X对应的index
        delta = zeros(1, length(w));                % 梯度
        for i = 1 : length(Y)
            tmpX = X(cnt : cnt + length(Y{i}) - 1, :);
            product = tmpX * w;                     % 叉乘
            tmpexp1 = exp(product);
            tmpexp2 = exp(Y{i});
            totalexp1 = sum(tmpexp1);
            totalexp2 = sum(tmpexp2);
            tmpresult1 = tmpexp1' * tmpX;   % 按照公式计算exp与X的乘积
            tmpresult2 = tmpexp2' * tmpX;

            % 迭代计算w中每一维的梯度
            for inx = 1 : length(w)
                delta(inx) = delta(inx) + tmpresult1(inx) / totalexp1 - tmpresult2(inx) / totalexp2;
            end
            cnt = cnt + length(Y{i});
        end

        w = w - delta' .* rate;     % 更新w
        if (mod(loop, times) == 0)  % 如果到了该记录的次数，就记录当前的w
            param(:, in) = w;
            in = in + 1;
        end
    end

    % 计算在vali上的NDCG，以选择最佳w
    [Xt,Yt] = read_letor([dname '/vali.txt']);
    nd = zeros(T / times, 10);  % 记录对应参数算出来的NDCG
    % 对于每个候选参数
    for i = 1 : T / times
        NDCG = zeros(1, 10);
        cnt = 0;
        % 对于每一类查询
        for j = 1 : length(Yt)
            tmpX = Xt(cnt + 1 : cnt + length(Yt{j}), :);    % 取出对应X
            YY = tmpX * param(:, i);                        % 算出当前模型对X的打分

            % 这里的处理是考虑到有些数据集上的某一类
            % 查询的样本数并没有10个的情况
            if (length(Yt{j}) < 10)
                size = length(Yt{j});
            else
                size = 10;
            end
            
            [Ys, ~] = sort(Yt{j}, 'descend');   % 根据样本原来的分值进行排序
            [~, index] = sort(YY, 'descend');   % 根据当前模型算出来的分值进行排序
            
            YYt = zeros(1, size);
            % 取出当前模型分值最大的size个样本对应的原来的Y值
            for k = 1 : size
                YYt(k) = Yt{j}(index(k));
            end

            NDCG = NDCG + calNDCG(Ys, YYt, size);   % 累加每类查询的NDCG
            cnt = cnt + length(Yt{j});              % 更新index
        end

        NDCG = NDCG ./ length(Yt);  % 算均值
        nd(i, :) = NDCG;            % 得出每个模型的NDCG的均值
    end

    % 计算每个位置的NDCG的总和，然后取最大的NDCG对应的w作为最佳模型
    avg = sum(nd, 2);
    [~, tin] = max(avg);
    w = param(:, tin);
    NDCGVA(fold, :) = nd(tin, :);   % 把该w作为该文件夹的模型
    
    % 把得到的w计算在test上的NDCG
    [Xt,Yt] = read_letor([dname '/test.txt']);
    %name = [dname 'sorce.txt'];
    %out = Xt * w;
    %save(name, 'out', '-ascii');
    %system(['perl Eval-Score-3.0.pl ' dname '/test.txt ' name ' ' dname '/ndcg.txt 0']);
    cnt = 0;
    NDCG = zeros(1, 10);
    for i = 1 : length(Yt)
        tmpX = Xt(cnt + 1 : cnt + length(Yt{i}), :);
        YY = tmpX * w;

        [Ys, ~] = sort(Yt{i}, 'descend');
        [~, index] = sort(YY, 'descend');
        if (length(Yt{i}) < 10)
            size = length(Yt{i});
        else
            size = 10;
        end
        YYt = zeros(1, size);
        for j = 1 : size
            YYt(j) = Yt{i}(index(j));
        end

        NDCG = NDCG + calNDCG(Ys, YYt, size);
        cnt = cnt + length(Yt{i});
    end

    NDCG = NDCG ./ length(Yt);
    NDCGTE(fold, :) = NDCG;
    
    % 计算在train上的NDCG
    cnt = 0;
    NDCG = zeros(1, 10);
    for i = 1 : length(Y)
        tmpX = X(cnt + 1 : cnt + length(Y{i}), :);
        YY = tmpX * w;

        [Ys, ~] = sort(Y{i}, 'descend');
        [~, index] = sort(YY, 'descend');
        if (length(Y{i}) < 10)
            size = length(Y{i});
        else
            size = 10;
        end
        YYt = zeros(1, size);
        for j = 1 : size
            YYt(j) = Y{i}(index(j));
        end

        NDCG = NDCG + calNDCG(Ys, YYt, size);
        cnt = cnt + length(Y{i});
    end

    NDCG = NDCG ./ length(Y);
    NDCGTR(fold, :) = NDCG;
end

% 输出到文件夹中
NDCGALL = {NDCGTE, NDCGVA, NDCGTR}; 
f = fopen(outfile, 'w');
for i = 1 : 3
    if (i == 1)
        fname = 'testing';
    elseif (i == 2)
        fname = 'validation';
    else
        fname = 'training';
    end
    fprintf(f, 'Performance on %s set\r\n', fname);
    fprintf(f, 'Folds	NDCG@1	NDCG@2	NDCG@3	NDCG@4	NDCG@5	NDCG@6	NDCG@7	NDCG@8	NDCG@9	NDCG@10\r\n');
    for j = 1 : 5
        fprintf(f, 'Fold%d   ', j);
        for k = 1 : 10
            fprintf(f, '%.4f  ', NDCGALL{i}(j, k));
        end
        fprintf(f, '\r\n');
    end
    fprintf(f, 'aver    ');
    avg = sum(NDCGALL{i}, 1);
    for j = 1 : 10
        fprintf(f, '%.4f  ', avg(j) / 5);
    end
    fprintf(f, '\r\n');
    fprintf(f, '\r\n');
end
fclose(f);

function [X, Y] = read_letor(filename)
    f = fopen(filename);
    X = zeros(2e5, 0);
    qid = '';
    i = 0;
    q = 0;

    while 1
	l = fgetl(f);
	if ~ischar(l)
	    break;
	end

	i = i + 1;
	[lab,  ~, ~, ind] = sscanf(l, '%d qid:', 1); 
	l(1:ind-1) = [];	
	[nqid, ~, ~, ind] = sscanf(l, '%s', 1); 
	l(1:ind-1 )= []; 

	if ~strcmp(nqid, qid)
	    q = q + 1;
	    qid = nqid;
	    Y{q} = lab;
	else
	    Y{q} = [Y{q}; lab];
	end

	tmp = sscanf(l, '%d:%f'); 
	X(i, tmp(1 : 2 : end)) = tmp(2 : 2 : end);
    end
    
    X = X(1 : i, :);
    fclose(f);
	       