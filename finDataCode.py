import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

matplotlib.use('Agg')

#1 Data Preprocessing
# Load the dataset
file_path = "multi_stock_data.csv"  # Change this to your actual file path
data = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)

# Display the original data sample
print("Original Data Sample:")
print(data.head())

# Handle missing values by forward-filling
data = data.ffill()

# Display the preprocessed data
print("\nPreprocessed Data Sample:")
print(data.head())


#2 Sorting the Data Using Merge Sort
def merge_sort(data):
    if len(data) > 1:
        mid = len(data) // 2
        left_half = data[:mid]
        right_half = data[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                data[k] = left_half[i]
                i += 1
            else:
                data[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            data[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            data[k] = right_half[j]
            j += 1
            k += 1


# Convert the DataFrame to a list of timestamps for sorting
sorted_data_list = list(data.index.to_list())

# Sort the data using the merge sort implementation
merge_sort(sorted_data_list)

# Reindex the DataFrame to the sorted order
sorted_data = data.reindex(sorted_data_list)

print("\nSorted Data Sample:")
print(sorted_data.head())

#3 Calculate Daily Changes for 2D Kadane's Algorithm
# Select two stocks for analysis (e.g., 'TSLA' and 'META')
stocks = ['TSLA', 'META']
price_changes_matrix = []

for stock in stocks:
    # Calculate daily changes as differences between consecutive days closing prices
    daily_changes = data[stock]['Close'].diff().fillna(0).values
    price_changes_matrix.append(daily_changes)

# Transpose the price_changes_matrix for 2D Kadane's input
price_changes_matrix = list(map(list, zip(*price_changes_matrix)))


#4 2D Kadane's Algorithm for Maximum Gain Detection
def kadane_2d_algorithm(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    max_sum = float('-inf')
    final_left = final_right = final_top = final_bottom = 0

    for left in range(num_cols):
        temp = [0] * num_rows

        for right in range(left, num_cols):
            # Update temporary array with sums of the current column range
            for i in range(num_rows):
                temp[i] += matrix[i][right]

            # Apply 1D Kadane's algorithm to find the maximum subarray sum for this column range
            current_sum, start_row, end_row = kadane_algorithm(temp)

            # Update maximum sum and boundaries if a new maximum is found
            if current_sum > max_sum:
                max_sum = current_sum
                final_left = left
                final_right = right
                final_top = start_row
                final_bottom = end_row

    return max_sum, final_left, final_right, final_top, final_bottom


def kadane_algorithm(array):
    max_current = max_global = array[0]
    start = end = s = 0

    for i in range(1, len(array)):
        if array[i] > max_current + array[i]:
            max_current = array[i]
            s = i
        else:
            max_current += array[i]

        if max_current > max_global:
            max_global = max_current
            start = s
            end = i

    return max_global, start, end


# Apply the 2D Kadane's algorithm to the matrix of daily price changes
max_gain, left, right, top, bottom = kadane_2d_algorithm(price_changes_matrix)

print(f"\nMaximum gain for the 2D analysis: {max_gain}")
print(f"Time Period: From index {left} to index {right}")
print(f"Stocks range: From row {top} to row {bottom}")

#5 Visualization of the Maximum Gain Period
plt.figure(figsize=(12, 6))

# Plot the selected stocks' closing prices
plt.plot(data.index, data['TSLA']['Close'], label='TSLA Close Prices', color='blue')
plt.plot(data.index, data['META']['Close'], label='META Close Prices', color='green')

# Highlight the detected 2D region of time range where the maximum gain occurs
if 0 <= left < len(data) and 0 <= right < len(data):
    plt.axvspan(data.index[left], data.index[right], color='red', alpha=0.3, label='Maximum Gain Period')

# Formatting the plot
plt.title('Maximum Gain Period Based on Daily Price Changes (2D Analysis for TSLA and META)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

# Save the plot as an image file
plt.savefig('C:/CMPSC 463/max_gain_2d_analysis_final.png')
plt.close()

print("The analysis is complete and the visualization has been saved as 'max_gain_2d_analysis_final.png'.")

# --- Step 5: Plotting Each Stock's Trend Separately ---
for stock in ['MSTR', 'META', 'NVDA', 'TSLA', 'SPY']:
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[stock]['Close'], label=f'{stock} Close Prices', color='blue')

    # Formatting the plot
    plt.title(f'{stock} Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    # Save the plot as a separate image file for each stock
    plt.savefig(f'C:/CMPSC 463/{stock}_trend.png')
    plt.close()

    print(f"The trend for {stock} has been saved as '{stock}_trend.png'.")
