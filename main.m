close all
clear
clc
%% Read Images

% the size of images must be equal

[file, pathname] = uigetfile('*.jpg','Load Image 1 ');cd(pathname);
a=imread(file);
[file, pathname] = uigetfile('*.jpg','Load Image 2 ');cd(pathname);
b=imread(file);


%%   Wavelet Transform 


[a1,b1,c1,d1]=dwt2(a,'db2');
[a2,b2,c2,d2]=dwt2(b,'db2');

[k1,k2]=size(a1);


%% Fusion Rules

%% Average Rule

for i=1:k1
    for j=1:k2
        a3(i,j)=(a1(i,j)+a2(i,j))/2;
   end
end

%% Max Rule


for i=1:k1
    for j=1:k2
        b3(i,j)=max(b1(i,j),b2(i,j));
        c3(i,j)=max(c1(i,j),c2(i,j));
        d3(i,j)=max(d1(i,j),d2(i,j));
    end
end


%% Inverse Wavelet Transform 

c=idwt2(a3,b3,c3,d3,'db2');
imshow(a)
title('First Image')
figure,imshow(b)
title('Second Image')
figure,imshow(c,[])
title('Fused Image')



%% Performance Criteria

CR1=corr2(a,c);
CR2=corr2(b,c);
S1=snrr(double(a),double(c));
S2=snrr(double(b),double(c));


fprintf('Correlation between first image and fused image =%f \n\n',CR1);
fprintf('Correlation between second image and fused image =%f \n\n',CR2);
fprintf('SNR between first image and fused image =%4.2f db\n\n',S1);
fprintf('SNR between second image and fused image =%4.2f db \n\n',S2);


