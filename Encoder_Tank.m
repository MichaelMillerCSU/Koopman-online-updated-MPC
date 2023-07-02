function y = Encoder_Tank(x)
    load('./Weights/Tank_New.mat');
    x1 = poslin(W1 * x + b1');
    x2 = poslin(W2 * x1 + b2');
    y = W3 * x2 + b3';
end

