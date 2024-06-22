% Parameters for water recycling and reuse
initial_water_usage = 100; % Initial water usage in million cubic meters
recycling_increase_rate = 0.04; % Annual increase in recycling rate (4%)

% Initialize water usage array
water_usage_recycling = zeros(size(years));
water_usage_recycling(1) = initial_water_usage;

% Simulate water usage reduction due to recycling and reuse
for t = 2:length(years)
    recycling_factor = (1 - recycling_increase_rate)^(years(t));
    water_usage_recycling(t) = initial_water_usage * recycling_factor;
end

% Plot the simulation
figure;
plot(years, water_usage_recycling, 'c-', 'LineWidth', 2);
xlabel('Years');
ylabel('Water Usage (million cubic meters)');
title('Water Usage Reduction with Recycling and Reuse');
grid on;

% Save the plot as an image
saveas(gcf, 'water_usage_recycling.png');
