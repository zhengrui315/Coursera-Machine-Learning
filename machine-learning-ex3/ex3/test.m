for i = 1:3,
for j = 1:4,
if i != j,
fprintf(' %d\n', i^j);
else,
fprintf('equal\n');
endif;
end;
end;