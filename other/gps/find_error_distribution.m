function [distribution] = find_error_distribution(error, incriment)

error = unique(error, 'sorted');

num_inc = ceil((max(error)-min(error)) / incriment);
incriments = linspace(min(error),max(error), num_inc);
err_dist = zeros(1, num_inc);
err_idx = 1;
for idx = 1:num_inc
    count = 1;
        while(error(err_idx) < incriments(idx) && err_idx < numel(error))
            count = count + 1;
            err_idx = err_idx + 1;
        end 
    err_dist(idx) = count;
end 

distribution = err_dist;

end