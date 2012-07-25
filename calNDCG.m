function [NDCG] = calNDCG(Y, Yt, size)
	n = 10;
	y = zeros(n,1);
	yt = zeros(n,1);
	LOG = ones(1,n);
	DCGy = zeros(1,n);
    DCGyt = zeros(1,n);
	%NOR = zeros(1,n);
	NDCG = zeros(1,n);
	for i = 1:size
		y(i) = 2^Y(i) -1;
		yt(i) = 2^Yt(i) -1;
		LOG(i) = 1/log2(i+1);
		
 		if(i==1)
			DCGy(i) = y(i) * LOG(i);
 			DCGyt(i) = yt(i) * LOG(i);
		else 
             DCGy(i) = DCGy(i-1) +y(i)*LOG(i);
             DCGyt(i) = DCGyt(i-1) +yt(i)*LOG(i);
         end;
         
        % NOR(i) = 1/DCGy(i);
 		
         if(Y(1)==0)
             NDCG(i) = 1;
         else
 		NDCG(i) = DCGyt(i)/DCGy(i);
         end;
    end
