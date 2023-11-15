function y = Encoder_One_Dimensional_System(x)
    load('./One_Dimensional_System22.mat');
    x1 = poslin(W1 * x + b1');
    x2 = poslin(W2 * x1 + b2');
    x3 = poslin(W3 * x2 + b3');
    y = W4 * x3 + b4';
end

