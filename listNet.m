function listNet(dataset)
NDCGTR = zeros(5, 10); % ��¼ÿ���ļ���train��NDCG
NDCGVA = zeros(5, 10); % ��¼ÿ���ļ���vali��NDCG
NDCGTE = zeros(5, 10); % ��¼ÿ���ļ���test��NDCG
outfile = 'out.txt';   % NDCG������ļ���
T = 500;               % ��������
times = 1;             % ÿ���ٴμ�¼һ��ģ�͵�w
rate = 0.01;           % ����

% ��ÿ���ļ��н��д���
for fold = 1 : 5
    dname = [dataset '/Fold' num2str(fold) '/'];    % train���ļ���
    [X, Y] = read_letor([dname '/train.txt']);      % ��������
    w = zeros(length(X(1, :)), 1);                  % ��ʼ��w��ȫΪ0
    param = zeros(length(w), T / times);            % ���ÿT/times�ε�w
    in = 1;                                         % param��ǰ��index
    
    % ����T��
    for loop = 1 : T
        cnt = 1;                                    % X��Ӧ��index
        delta = zeros(1, length(w));                % �ݶ�
        for i = 1 : length(Y)
            tmpX = X(cnt : cnt + length(Y{i}) - 1, :);
            product = tmpX * w;                     % ���
            tmpexp1 = exp(product);
            tmpexp2 = exp(Y{i});
            totalexp1 = sum(tmpexp1);
            totalexp2 = sum(tmpexp2);
            tmpresult1 = tmpexp1' * tmpX;   % ���չ�ʽ����exp��X�ĳ˻�
            tmpresult2 = tmpexp2' * tmpX;

            % ��������w��ÿһά���ݶ�
            for inx = 1 : length(w)
                delta(inx) = delta(inx) + tmpresult1(inx) / totalexp1 - tmpresult2(inx) / totalexp2;
            end
            cnt = cnt + length(Y{i});
        end

        w = w - delta' .* rate;     % ����w
        if (mod(loop, times) == 0)  % ������˸ü�¼�Ĵ������ͼ�¼��ǰ��w
            param(:, in) = w;
            in = in + 1;
        end
    end

    % ������vali�ϵ�NDCG����ѡ�����w
    [Xt,Yt] = read_letor([dname '/vali.txt']);
    nd = zeros(T / times, 10);  % ��¼��Ӧ�����������NDCG
    % ����ÿ����ѡ����
    for i = 1 : T / times
        NDCG = zeros(1, 10);
        cnt = 0;
        % ����ÿһ���ѯ
        for j = 1 : length(Yt)
            tmpX = Xt(cnt + 1 : cnt + length(Yt{j}), :);    % ȡ����ӦX
            YY = tmpX * param(:, i);                        % �����ǰģ�Ͷ�X�Ĵ��

            % ����Ĵ����ǿ��ǵ���Щ���ݼ��ϵ�ĳһ��
            % ��ѯ����������û��10�������
            if (length(Yt{j}) < 10)
                size = length(Yt{j});
            else
                size = 10;
            end
            
            [Ys, ~] = sort(Yt{j}, 'descend');   % ��������ԭ���ķ�ֵ��������
            [~, index] = sort(YY, 'descend');   % ���ݵ�ǰģ��������ķ�ֵ��������
            
            YYt = zeros(1, size);
            % ȡ����ǰģ�ͷ�ֵ����size��������Ӧ��ԭ����Yֵ
            for k = 1 : size
                YYt(k) = Yt{j}(index(k));
            end

            NDCG = NDCG + calNDCG(Ys, YYt, size);   % �ۼ�ÿ���ѯ��NDCG
            cnt = cnt + length(Yt{j});              % ����index
        end

        NDCG = NDCG ./ length(Yt);  % ���ֵ
        nd(i, :) = NDCG;            % �ó�ÿ��ģ�͵�NDCG�ľ�ֵ
    end

    % ����ÿ��λ�õ�NDCG���ܺͣ�Ȼ��ȡ����NDCG��Ӧ��w��Ϊ���ģ��
    avg = sum(nd, 2);
    [~, tin] = max(avg);
    w = param(:, tin);
    NDCGVA(fold, :) = nd(tin, :);   % �Ѹ�w��Ϊ���ļ��е�ģ��
    
    % �ѵõ���w������test�ϵ�NDCG
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
    
    % ������train�ϵ�NDCG
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

% ������ļ�����
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
	       