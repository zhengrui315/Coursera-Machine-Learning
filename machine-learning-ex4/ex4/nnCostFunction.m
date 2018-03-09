function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
a1 = [ones(1,m);X'];
%dim = (n+1)*m = (s1+1)*m
      
%dim(a1) = (s1+1) * m; dim(a2) = (s2+1)*m; dim(a3) = s3 *m
      
a2 = [ones(1,m);sigmoid(Theta1 * a1)]; %dim =(s2+1)*m
a3 = sigmoid(Theta2 * a2);%dim = s3*m
% a3 = h_theta has dimension: K * m
yy = [];
      
% make yy to be K * m = s3 * m
for i = 1:num_labels,
yy = [yy;y'==i];
end;
      
      theta2 = Theta2;
      theta2(:,1)=0;
      theta1 = Theta1;
      theta1(:,1)=0;
      
J = -1/m * trace(yy' * log(a3) + (1- yy') * log(1-a3)) + lambda/(2*m) * trace(theta1 * theta1') + lambda/(2*m) * trace(theta2 * theta2');
             
      
      
%Theta1_grad = [];
%Theta2_grad = zeros(size(Theta2));

%dim(d3) = s3 * m; dim(d2) = s2 * m;
      
delta3 = (a3 - yy);
delta2 = [];
Theta2p = Theta2(:,2:end)';
      
for i = 1:m,
delta2(:,i) = ((Theta2p * delta3)(:,i)) .* a2(2:end,i) .* (1 - a2(2:end,i));
end;
                     

                     
Theta2_grad = lambda/m * theta2 + 1/m * delta3 * a2';
Theta1_grad = lambda/m * theta1 + 1/m * delta2 * a1';

      
%fprintf('the dimension of delta3 is %i * %i \n',size(delta3,1),size(delta3,2));
%fprintf('the dimension of delta2 is %i * %i \n',size(delta2,1),size(delta2,2));
%fprintf('the dimension of theta2 is %i * %i \n',size(theta2,1),size(theta2,2));
%fprintf('the dimension of theta1 is %i * %i \n',size(theta1,1),size(theta1,2));
%fprintf('the dimension of delta3 is %i * %i \n',size(delta3,1),size(delta3,2));


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
