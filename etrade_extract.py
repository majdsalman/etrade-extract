import os
import warnings

import PyPDF2
import sys
from datetime import datetime, timedelta
import re
import holidays
import numpy as np
import pandas as pd

red_text_start = "\033[91m"
red_text_end = "\033[0m"


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text_content = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text_content += page.extract_text().lower()
        return text_content


def extract_value(regex, text, group=1):
    matches = re.search(regex, text)
    return matches.group(group) if matches else None


def get_next_weekday(date: datetime) -> datetime:
    swedish_holidays = holidays.Sweden()
    # Increment the date until it is a weekday and not a holiday
    while date.weekday() >= 5 or date in swedish_holidays:
        date += timedelta(days=1)
    return date


def extract_values_payslip(payslip_file):
    print(f"Reading payslip: {payslip_file}")
    text = extract_text_from_pdf(payslip_file)
    # Define a regex pattern to match the start and end dates
    date_pattern = r"(\d{8}) - (\d{8})"

    # Search for the pattern in the text
    start_date = datetime.strptime(extract_value(date_pattern, text, 1), '%Y%m%d')
    end_date = datetime.strptime(extract_value(date_pattern, text, 2), '%Y%m%d')

    # Updated regex pattern to capture the alphanumeric sequence and the numbers
    pattern = r"(e\d+)\s([\d\s]+,\d{2}) \*"

    # Use re.findall to extract all matches of the pattern
    matches = re.findall(pattern, text)

    # Process the matches to extract the alphanumeric sequence and convert the numbers to floats
    extracted_data = [{
        "id": match[0],
        "From Payslip": float(match[1].replace('\xa0', '').replace(' ', '').replace(',', '.')),
        "start_date": start_date,
        "end_date": end_date
    } for match in matches]

    return pd.DataFrame(extracted_data)


def read_benefit_history_rsu(benefit_history_file):
    print(f"Reading benefit history {benefit_history_file}")
    columns_to_read = ['Record Type', 'Grant Number', 'Date', 'Event Type', 'Qty. or Amount', 'Vest Period',
                       'Vest Date', 'Released Qty', 'Taxable Gain']
    df = pd.read_excel(benefit_history_file, usecols=columns_to_read, sheet_name=0)

    # Filter out rows where 'Vest Period' is NaN
    vesting_df = df[df['Vest Period'].notna()].copy()

    # Set 'Record Type' to "Vesting" for the filtered DataFrame
    vesting_df['Record Type'] = 'Vesting'

    # Group by 'Grant Number' and 'Vest Period' and aggregate the other columns
    FMV_df = vesting_df.groupby(['Grant Number', 'Vest Period']).agg({

        'Vest Date': 'first',  # Same for 'Vest Date'
        'Released Qty': 'sum',  # Sum the 'Released Qty'
        'Taxable Gain': 'sum',  # Sum the 'Taxable Gain'
    }).reset_index()
    FMV_df['FMV'] = np.where(FMV_df['Released Qty'] != 0,
                             FMV_df['Taxable Gain'] / FMV_df['Released Qty'],
                             np.nan)

    vesting_df = df[df['Event Type'] == "Shares released"]
    # Specify the columns you want to include in the new DataFrame
    columns_to_include = ['Grant Number', 'Date', 'Event Type', 'Qty. or Amount']

    vesting_df = vesting_df[columns_to_include]

    # Merge the two DataFrames on 'Grant Number' and 'Date'
    result_df = pd.merge(vesting_df, FMV_df, left_on=['Grant Number', 'Date'],
                         right_on=['Grant Number', 'Vest Date'])
    result_df['Date'] = pd.to_datetime(result_df['Date'])
    # Sort the DataFrame by the 'Date' column in ascending order
    result_df = result_df.sort_values(by='Date', ascending=True)
    columns_to_include = ['Grant Number', 'Date', 'Event Type', 'Qty. or Amount', 'FMV']

    return result_df[columns_to_include]


def read_benefit_history_options(benefit_history_file):
    print(f"Reading benefit history {benefit_history_file}")
    columns_to_read = ['Grant Number', 'Event Type', 'Date', 'Qty']
    df = pd.read_excel(benefit_history_file, usecols=columns_to_read, sheet_name=1)
    df = df[df['Event Type'].isin(['Shares sold', 'Shares exercised'])]
    df['Date'] = pd.to_datetime(df['Date'])
    # df = df.groupby(['Grant Number', 'Date', 'Event Type'])['Qty'].sum().reset_index()
    df = df.sort_values(by='Date')
    return df


def read_gain_loss(gain_loss_file):
    print(f"Reading trades: {gain_loss_file}")
    columns_to_read = ['Record Type', 'Qty.', 'Date Sold', 'Adjusted Cost Basis Per Share', 'Gain/Loss',
                       'Adjusted Gain/Loss', 'Order Type', 'Grant Price', 'Proceeds Per Share',
                       'Order Number', 'Grant Number', 'Exercise Date FMV', 'Date Acquired', 'Exercise Date FMV']
    df = pd.read_excel(gain_loss_file, usecols=columns_to_read)
    df['Date Sold'] = pd.to_datetime(df['Date Sold'])
    df['Date Acquired'] = pd.to_datetime(df['Date Acquired'])
    df = df.groupby(
        ['Order Type', 'Order Number', 'Date Sold', 'Grant Number', 'Exercise Date FMV', 'Date Acquired']).agg({
        'Record Type': 'first',  # Same for 'Vest Date'
        'Qty.': 'sum',  # Sum the 'Released Qty
        'Adjusted Cost Basis Per Share': 'first',
        'Gain/Loss': 'sum',
        'Adjusted Gain/Loss': 'sum',
        'Grant Price': 'first',
        'Proceeds Per Share': 'mean',
        # '
    }).reset_index()
    df.loc[df['Grant Price'] == 0, 'Grant Price'] = np.nan

    return df


def match_orders_payslip(orders_df, benefits_df):
    # Assuming orders_df and benefits_df are your existing DataFrames
    # Convert the 'Date Sold' column to datetime for proper sorting
    orders_df['Date Sold'] = pd.to_datetime(orders_df['Date Sold'])
    if benefits_df.empty:
        orders_df['From Payslip'] = np.nan
        return orders_df
    # Filter for 'Same-Day Sale' order types
    same_day_sales_df = orders_df[orders_df['Order Type'] == 'Same-Day Sale'].copy()

    # Extract the year from 'Date Sold' and 'start_date'
    same_day_sales_df['Year'] = same_day_sales_df['Date Sold'].dt.year
    benefits_df['Year'] = pd.to_datetime(benefits_df['start_date']).dt.year

    # Group by year and get the counts
    same_day_sales_counts = same_day_sales_df.groupby('Year').size()
    benefits_counts = benefits_df.groupby('Year').size()

    # Align both Series on their index (Year) and fill missing values with 0
    aligned_same_day_sales_counts, aligned_benefits_counts = same_day_sales_counts.align(benefits_counts, fill_value=0)

    # Check for any years that do not match and issue a warning
    non_matching_years = aligned_same_day_sales_counts[aligned_same_day_sales_counts != aligned_benefits_counts].index
    if not non_matching_years.empty:
        # ANSI escape code for red text

        print(
            f"{red_text_start}The number of elements for the following years does not match between orders and benefits: {list(non_matching_years)}{red_text_end}")
        print(
            f"{red_text_start}Please fill in payslip data manually or add payslips and orders for all years{red_text_end}")

    # Now you can safely compare the two Series for their common years
    # This will give you a boolean Series indicating where the counts match
    matching_counts = aligned_same_day_sales_counts == aligned_benefits_counts

    # Filter the DataFrames to only include the years where the counts match
    matching_years = matching_counts[matching_counts].index
    filtered_same_day_sales_df = same_day_sales_df[same_day_sales_df['Year'].isin(matching_years)]
    filtered_benefits_df = benefits_df[benefits_df['Year'].isin(matching_years)]

    # Sort the filtered DataFrames
    filtered_same_day_sales_df = filtered_same_day_sales_df.sort_values(by='Order Number')
    filtered_benefits_df = filtered_benefits_df.sort_values(by='id')

    # Reset index to ensure proper alignment
    filtered_same_day_sales_df = filtered_same_day_sales_df.reset_index(drop=True)
    filtered_benefits_df = filtered_benefits_df.reset_index(drop=True)

    # Assign the 'total' column from filtered_benefits_df to filtered_same_day_sales_df
    filtered_same_day_sales_df['From Payslip'] = filtered_benefits_df['From Payslip']

    # Now, if you want to update the original orders_df with the new 'Total' values, you can merge them back
    orders_df = pd.merge(orders_df, filtered_same_day_sales_df[['Order Number', 'From Payslip']], on='Order Number',
                         how='left')

    # The 'Total' column will now be in the orders_df with NaN for rows that are not 'Same-Day Sale'
    # or if the number of elements did not match for a particular year
    return orders_df


def combine_data(trades_df, benefits_rsu_df, benefits_options_df):
    rows_list = []

    for index, row in trades_df.iterrows():
        if row["Order Type"] == 'Same-Day Sale':
            fmv = row['Adjusted Cost Basis Per Share']
            net_proceeds = row['Gain/Loss']
        elif row["Order Type"] == 'Sell Restricted Stock':
            fmv = row['Proceeds Per Share']
            net_proceeds = row['Qty.'] * fmv
        elif row["Order Type"] == 'Sell Exercised Stock':
            fmv = row['Proceeds Per Share']  # row['Exercise Date FMV']
            net_proceeds = np.nan
        else:
            fmv = np.nan
            net_proceeds = np.nan

        new_row = {
            'Date': row['Date Sold'],  # Example transformation
            'FX Date': get_next_weekday(row['Date Sold']),
            'Operation': row['Order Type'],
            'Qty': row['Qty.'],
            'FMV': fmv,
            'Fee': np.nan,  # row['Adjusted Gain/Loss'] if row['Order Type'] == 'Same-Day Sale' else
            'Grant price': row['Grant Price'],
            'Net proceeds': net_proceeds,
            'Income declared on payslip': row['From Payslip'],
            'Grant': row['Grant Number'],
            'Date Acquired': row['Date Acquired'],
            'Exercise Date FMV': row['Exercise Date FMV'] if row['Order Type'] == 'Sell Exercised Stock' else np.nan,
            'Order Number': row['Order Number']
        }
        # Append the transformed row to the new DataFrame
        rows_list.append(new_row)

    for index, row in benefits_rsu_df.iterrows():
        new_row = {
            'Date': row['Date'],  # Example transformation
            'FX Date': get_next_weekday(row['Date']),
            'Operation': row['Event Type'],
            'Qty': row['Qty. or Amount'],
            'FMV': row['FMV'],
            'Fee': np.nan,
            'Grant price': np.nan,
            'Net proceeds': np.nan,
            'Income declared on payslip': np.nan,
            'Grant': row['Grant Number'],
            'Date Acquired': np.nan,
            'Exercise Date FMV': np.nan,
            'Order Number': np.nan
        }
        # Append the transformed row to the new DataFrame
        rows_list.append(new_row)
    new_df = pd.DataFrame(rows_list)

    # Sanity check options benefits
    # Assuming trades_df and options_benefits_df are your two dataframes

    # Merge the dataframes on the specified columns with an indicator to track the merge status
    merged_df = pd.merge(new_df, benefits_options_df, left_on=['Grant', 'Qty', 'Date'],
                         right_on=['Grant Number', 'Qty', 'Date'], how='left', indicator=True)

    # Filter the merged dataframe for "Same-Day Sale" operations
    same_day_sales = merged_df[merged_df['Operation'] == 'Same-Day Sale']

    # Check if there are any trades that were not matched in the benefits df
    unmatched_trades = same_day_sales[same_day_sales['_merge'] == 'left_only']

    if len(unmatched_trades) > 0:
        print(
            f"{red_text_start}Unmatched same day options trades{red_text_end}")
        print(unmatched_trades[['Date', 'Operation', 'Qty', 'Grant']].all)

    # Match same day sales and add sales rows to Same day sales
    same_day_sales = new_df[new_df['Operation'] == 'Same-Day Sale']

    # Create a new dataframe with modified rows
    new_rows = same_day_sales.copy()
    new_rows['Qty'] = -new_rows['Qty']
    new_rows['Operation'] = 'Sell'

    # Concatenate the new rows with the original trades dataframe
    new_df = pd.concat([new_df, new_rows], ignore_index=True)

    # Remove the matching "Same-Day Sale" rows from the benefits dataframe
    matching_benefits = benefits_options_df[benefits_options_df.set_index(['Grant Number', 'Qty', 'Date']).index.isin(
        same_day_sales.set_index(['Grant', 'Qty', 'Date']).index)]
    benefits_options_df = benefits_options_df.drop(matching_benefits.index)

    #### Match Exercised options with Sell Exercised Stock
    # Filter the trades dataframe for "Sell Exercised Stock" operations
    sell_exercised_stock = new_df[new_df['Operation'] == 'Sell Exercised Stock']

    # Merge with the benefits dataframe where "Event Type" is "Shares exercised"
    merged_df = pd.merge(sell_exercised_stock,
                         benefits_options_df[benefits_options_df['Event Type'] == 'Shares exercised'],
                         left_on=['Grant', 'Qty', 'Date Acquired'], right_on=['Grant Number', 'Qty', 'Date'],
                         suffixes=('', '_benefits'), indicator=True)

    # Create a new dataframe with the "Exercise" operation
    exercise_rows = merged_df.copy()

    exercise_rows['Operation'] = 'Exercise'
    exercise_rows['FMV'] = exercise_rows['Exercise Date FMV']
    exercise_rows['Date'] = exercise_rows['Date Acquired']
    exercise_rows['FX Date'] = exercise_rows['Date'].apply(get_next_weekday)
    exercise_rows.rename(columns={'Date': 'Date'}, inplace=True)

    # Select only the relevant columns from the original trades dataframe
    exercise_rows = exercise_rows[new_df.columns]

    # Concatenate the new "Exercise" rows with the original trades dataframe
    new_trades_df = pd.concat([new_df, exercise_rows], ignore_index=True)

    shares_exercised = benefits_options_df[benefits_options_df['Event Type'] == 'Shares exercised']

    # Remove the matching rows from the filtered dataframe
    shares_exercised = shares_exercised[~shares_exercised.set_index(['Grant Number', 'Qty', 'Date'])
    .index.isin(exercise_rows.set_index(['Grant', 'Qty', 'Date Acquired']).index)]

    # Now, recombine the filtered "Shares exercised" rows with the other rows from the original benefits dataframe
    benefits_options_df = shares_exercised

    rows_list = []
    for index, row in benefits_options_df.iterrows():
        new_row = {
            'Date': row['Date'],  # Example transformation
            'FX Date': get_next_weekday(row['Date']),
            'Operation': 'UNMATCHED EXERCISE',
            'Qty': row['Qty'],
            'Grant': row['Grant Number'],
            'Date Acquired': row['Date']
        }
        # Append the transformed row to the new DataFrame
        rows_list.append(new_row)
    unmatched_options_df = pd.DataFrame(rows_list)

    # Ensure that the new DataFrame has the same columns as new_trades_df, adding any missing columns with NaN values
    for column in new_trades_df.columns:
        if column not in unmatched_options_df.columns:
            unmatched_options_df[column] = np.nan

    # Reorder the columns to match new_trades_df
    unmatched_options_df = unmatched_options_df.reindex(columns=new_trades_df.columns)

    # Explicitly set the data types for the new DataFrame to match the original DataFrame's column data types
    for column in unmatched_options_df.columns:
        if unmatched_options_df[column].dtype != new_trades_df[column].dtype:
            unmatched_options_df[column] = unmatched_options_df[column].astype(new_trades_df[column].dtype)

    # Concatenate the dataframes
    new_trades_df = pd.concat([new_trades_df, unmatched_options_df], ignore_index=True)

    ### Group RSU trades
    sell_restricted_stock_df = new_trades_df[new_trades_df['Operation'] == 'Sell Restricted Stock']
    aggregated_df = sell_restricted_stock_df.groupby(['Date', 'FX Date', 'Operation', 'Order Number']).agg({
        'Qty': 'sum',
        'FMV': 'mean',
        'Grant price': 'mean',
        'Net proceeds': 'sum',
        'Income declared on payslip': 'sum',
        'Grant': 'first',
        'Date Acquired': 'first',
        'Exercise Date FMV': 'first'
    }).reset_index()

    # Remove original "Sell Restricted Stock" rows from the original dataframe
    df_without_sell_restricted_stock = new_trades_df[new_trades_df['Operation'] != 'Sell Restricted Stock']

    # Concatenate the original dataframe with the aggregated dataframe
    new_trades_df = pd.concat([df_without_sell_restricted_stock, aggregated_df], ignore_index=True)

    ## Cleanup data
    # Define a mapping of old operation values to new operation values
    operation_mapping = {
        'Sell Restricted Stock': 'Sell',
        'Sell Exercised Stock': 'Sell',
        'Same-Day Sale': 'Exercise',
        'Shares released': 'RSU vesting'
    }

    # Replace the values in the Operation column based on the mapping
    new_trades_df['Operation'] = new_trades_df['Operation'].replace(operation_mapping)

    # set negative Qty
    new_trades_df.loc[(new_trades_df['Operation'] == 'Sell') & (new_trades_df['Qty'] > 0), 'Qty'] *= -1
    new_trades_df.loc[new_trades_df['Operation'] != 'Exercise', ['Net proceeds', 'Grant price',
                                                                 'Income declared on payslip']] = np.nan

    new_trades_df = new_trades_df.sort_values(by=['Date', 'Operation', 'Qty'], ignore_index=True)
    return new_trades_df


def save_files(df_all):
    date_columns = ['Date', 'FX Date', 'Date Acquired']  # List of your date columns
    for date_column in date_columns:
        if date_column in df_all.columns and pd.api.types.is_datetime64_any_dtype(df_all[date_column]):
            df_all[date_column] = df_all[date_column].dt.strftime('%d/%m/%Y')  # Format as 'YYYY-MM-DD'

    df_all.to_csv('transactions.csv', index=False)
    output_file_name = 'transactions.xlsx'

    # Create a Pandas Excel writer using XlsxWriter as the engine
    with pd.ExcelWriter(output_file_name, engine='xlsxwriter') as writer:
        # Write the dataframe data to XlsxWriter
        df_all.to_excel(writer, index=False, sheet_name='trades_and_benefits')

        workbook = writer.book

        worksheet = writer.sheets['trades_and_benefits']

        green_fill = workbook.add_format({'bg_color': '#C6EFCE'})
        green_number_format = workbook.add_format({'bg_color': '#C6EFCE', 'num_format': '0.000'})
        red_fill = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        num_rows = len(df_all.index)

        for column in ['A', 'B', 'C', 'D']:
            worksheet.set_column(f'{column}2:{column}{num_rows + 1}', 15, green_fill)

        for column in ['E', 'F', 'G', 'H', 'I']:
            worksheet.set_column(f'{column}2:{column}{num_rows + 1}', 15, green_number_format)

        operation_col_idx = df_all.columns.get_loc('Operation') + 1  # 1-based index
        operation_col_letter = chr(operation_col_idx + 64)  # Convert to Excel column letter
        worksheet.conditional_format(f'{operation_col_letter}2:{operation_col_letter}{num_rows + 1}', {
            'type': 'text',
            'criteria': 'containing',
            'value': 'UNMATCHED EXERCISE',
            'format': red_fill
        })

    print()
    print(f"\n\033[1mSaved transactions file to {output_file_name}\n\033[0m")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python etrade_extract.py <path_to_transactions_folder>")

    transactions_folder = sys.argv[1]
    df_gains_losses = pd.DataFrame()
    df_payslip_options = pd.DataFrame()
    for root, dirs, files in os.walk(transactions_folder):
        for file in files:
            file_name = file.lower()
            file_path = os.path.join(root, file)
            if file_name.endswith('.xlsx'):
                if "benefit" in file_name:
                    df_benefits_rsu = read_benefit_history_rsu(file_path)
                    df_benefits_options = read_benefit_history_options(file_path)
                if "g&l" in file_name or "gain" in file_name:
                    df_gl = read_gain_loss(file_path)
                    df_gains_losses = pd.concat([df_gains_losses, df_gl], ignore_index=True)

            if file_name.endswith('.pdf'):
                if 'payslip' in file_name:
                    df_payslip = extract_values_payslip(file_path)
                    df_payslip_options = pd.concat([df_payslip_options, df_payslip], ignore_index=True)

    if df_gains_losses.empty:
        print(f"{red_text_start}No trade data found in dir: {transactions_folder}{red_text_end}")
        exit(1)

    df_orders = match_orders_payslip(df_gains_losses, df_payslip_options)
    df_all = combine_data(df_orders, df_benefits_rsu, df_benefits_options)
    operation_stats = df_all.groupby('Operation').agg(
        Operation_Count=('Operation', 'count'),
        Sum=('Qty', lambda x: x.sum())
    )

    # Display the statistics
    print("\n\033[1mDone!\033[0m")
    print(operation_stats)

    save_files(df_all)
