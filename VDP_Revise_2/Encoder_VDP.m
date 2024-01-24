function y = Encoder_VDP(x)
    load('Good_VDP.mat');
    x1 = poslin(W1 * x + b1');
    x2 = poslin(W2 * x1 + b2');
    x3 = poslin(W3 * x2 + b3');
    y = W4 * x3 + b4';
end

