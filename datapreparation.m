% Generate synthetic data for demonstration purposes
years = (2000:2050)';
temperature = linspace(20, 25, length(years))'; % Simulated temperature data
population = linspace(1, 2, length(years))'; % Simulated population data (in millions)

% Simulated water usage based on temperature and population
water_usage = 100 + 10 * temperature + 50 * population + randn(length(years), 1) * 5;

% Combine data into a table
data = table(years, temperature, population, water_usage);

% Split data into training and testing sets
train_ratio = 0.8;
num_train = round(train_ratio * length(data.years));
train_data = data(1:num_train, :);
test_data = data(num_train+1:end, :);

% Train a linear regression model
mdl = fitlm(train_data, 'water_usage ~ temperature + population');

% Predict water usage
predicted_usage = predict(mdl, test_data);

% Plot the results
figure;
plot(test_data.years, test_data.water_usage, 'b', 'DisplayName', 'Actual');
hold on;
plot(test_data.years, predicted_usage, 'r--', 'DisplayName', 'Predicted');
xlabel('Year');
ylabel('Water Usage (million cubic meters)');
title('Actual vs Predicted Water Usage');
legend;
grid on;

% Save the plot as an image
saveas(gcf, 'predictive_analytics.png');
