import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import re
warnings.filterwarnings('ignore')

def extract_trade_data(file_path):
    """Extract trade data from Excel workbook with multiple sheets"""
    
    # Read the Excel file
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    
    print(f"Total sheets found: {len(sheet_names)}")
    
    # Identify monthly sheets (exclude 'Data Calcs')
    monthly_sheets = [name for name in sheet_names if name != 'Data Calcs']
    print(f"Monthly trade sheets found: {len(monthly_sheets)} (Jan 2024 to Jan 2026)")
    print(f"Excluded sheet: 'Data Calcs' (calculation sheet)")
    
    all_data = []
    sheet_processing_summary = []
    
    for sheet_index, sheet_name in enumerate(monthly_sheets):
        try:
            print(f"\n{'='*80}")
            print(f"Processing sheet {sheet_index+1}/{len(monthly_sheets)}: {sheet_name}")
            print(f"{'='*80}")
            
            # Read the sheet
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            
            # Find the start of LIVE trade data
            live_header_row = None
            for idx, row in df.iterrows():
                if any(isinstance(cell, str) and "DATE" in str(cell).upper() for cell in row):
                    live_header_row = idx
                    break
            
            if live_header_row is None:
                print(f"  ❌ Error: No DATE header found in sheet {sheet_name}")
                continue
            
            # Get header row to understand column structure
            headers = df.iloc[live_header_row].tolist()
            print(f"  Header columns found: {[str(h)[:20] for h in headers if not pd.isna(h)]}")
            
            # Find where yellow box starts (end of LIVE trades)
            yellow_box_start = None
            yellow_box_text = "DO NOT EDIT THE FIELDS INSIDE THIS YELLOW BOX"
            for idx, row in df.iterrows():
                if idx > live_header_row:
                    for cell in row:
                        if isinstance(cell, str) and yellow_box_text in cell:
                            yellow_box_start = idx
                            break
                    if yellow_box_start:
                        break
            
            # Find DEMO section
            demo_section_start = None
            demo_section_text = "USE THE AREA BELOW FOR TRAINING PURPOSES USING A DEMO ACCOUNT"
            for idx, row in df.iterrows():
                if idx > live_header_row:
                    for cell in row:
                        if isinstance(cell, str) and demo_section_text in cell:
                            demo_section_start = idx
                            break
                    if demo_section_start:
                        break
            
            # Extract LIVE trades (between header_row+1 and yellow_box_start-1)
            live_trades = []
            if yellow_box_start:
                live_end_row = yellow_box_start
            elif demo_section_start:
                live_end_row = demo_section_start
            else:
                live_end_row = len(df)
            
            print(f"  Extracting LIVE trades from rows {live_header_row+1} to {live_end_row-1}")
            
            for i in range(live_header_row + 1, live_end_row):
                row_data = df.iloc[i].tolist()
                
                # Check if this is a valid trade row
                if (len(row_data) > 0 and 
                    not all(pd.isna(cell) for cell in row_data[:3]) and
                    not pd.isna(row_data[0]) and 
                    str(row_data[0]).strip() not in ['', 'nan', 'NaN', 'None'] and
                    not any(x in str(row_data[0]).upper() for x in ['DO NOT EDIT', 'USE THE AREA', 'TRADING RESULTS', 'TOTAL'])):
                    
                    # Ensure we have at least 8 columns, pad with empty strings
                    while len(row_data) < 8:
                        row_data.append('')
                    
                    # Add LIVE status
                    row_data.append('LIVE')
                    live_trades.append(row_data[:9])  # Take first 9 columns
            
            # Extract DEMO trades if demo section exists
            demo_trades = []
            if demo_section_start:
                # Find DEMO header
                demo_header_row = None
                for idx in range(demo_section_start + 1, min(demo_section_start + 10, len(df))):
                    row = df.iloc[idx].tolist()
                    if any(isinstance(cell, str) and "DATE" in str(cell).upper() for cell in row):
                        demo_header_row = idx
                        break
                
                if demo_header_row:
                    print(f"  Extracting DEMO trades from rows {demo_header_row+1} onward")
                    
                    # Find end of DEMO trades
                    demo_end_row = len(df)
                    for idx in range(demo_header_row + 1, len(df)):
                        row = df.iloc[idx].tolist()
                        row_text = ' '.join([str(cell) for cell in row if isinstance(cell, str)])
                        if "TRADING RESULTS" in row_text.upper():
                            demo_end_row = idx
                            break
                        if yellow_box_text in row_text:
                            demo_end_row = idx
                            break
                    
                    for i in range(demo_header_row + 1, demo_end_row):
                        row_data = df.iloc[i].tolist()
                        
                        # Check if valid trade row
                        if (len(row_data) > 0 and 
                            not all(pd.isna(cell) for cell in row_data[:3]) and
                            not pd.isna(row_data[0]) and 
                            str(row_data[0]).strip() not in ['', 'nan', 'NaN', 'None'] and
                            not any(x in str(row_data[0]).upper() for x in ['DO NOT EDIT', 'USE THE AREA', 'TRADING RESULTS', 'TOTAL'])):
                            
                            # Ensure we have at least 8 columns, pad with empty strings
                            while len(row_data) < 8:
                                row_data.append('')
                            
                            # Add DEMO status
                            row_data.append('DEMO')
                            demo_trades.append(row_data[:9])
            
            # Try to find "Total Percentage Profit of Current Month" from source
            source_total_percentage = None
            if yellow_box_start:
                for i in range(yellow_box_start, min(yellow_box_start + 20, len(df))):
                    row = df.iloc[i].tolist()
                    for j, cell in enumerate(row):
                        if isinstance(cell, str) and "Total Percentage Profit of Current Month" in cell:
                            if j + 1 < len(row):
                                source_total_percentage = row[j + 1]
                                break
                    if source_total_percentage is not None:
                        break
            
            # Calculate total percentage from extracted LIVE trades
            calculated_total_percentage = 0
            live_trade_count = 0
            demo_trade_count = 0
            
            for trade in live_trades:
                live_trade_count += 1
                if len(trade) > 6 and not pd.isna(trade[6]) and str(trade[6]).strip():
                    try:
                        calculated_total_percentage += float(trade[6])
                    except (ValueError, TypeError):
                        pass
            
            for trade in demo_trades:
                demo_trade_count += 1
            
            # Check for discrepancy
            discrepancy = None
            discrepancy_amount = None
            
            if source_total_percentage is not None and not pd.isna(source_total_percentage):
                try:
                    source_val = float(source_total_percentage)
                    calc_val = calculated_total_percentage
                    if abs(source_val - calc_val) > 0.001:
                        discrepancy_amount = source_val - calc_val
                        discrepancy = f"Discrepancy: {discrepancy_amount:.4f} (Source: {source_val:.4f}, Calculated: {calc_val:.4f})"
                except (ValueError, TypeError):
                    pass
            
            # Add to summary
            sheet_summary = {
                'Sheet_Name': sheet_name,
                'Live_Records': live_trade_count,
                'Demo_Records': demo_trade_count,
                'Source_Total_Percentage': round(float(source_total_percentage), 4) if source_total_percentage is not None and not pd.isna(source_total_percentage) else None,
                'Calculated_Total_Percentage': round(calculated_total_percentage, 4),
                'Discrepancy': discrepancy,
                'Discrepancy_Amount': discrepancy_amount
            }
            
            sheet_processing_summary.append(sheet_summary)
            
            # Print processing summary for this sheet
            print(f"  ✓ LIVE records found: {live_trade_count}")
            print(f"  ✓ DEMO records found: {demo_trade_count}")
            if source_total_percentage is not None:
                print(f"  ✓ Source 'Total Percentage Profit of Current Month': {source_total_percentage}")
            print(f"  ✓ Calculated total % from LIVE records: {calculated_total_percentage:.4f}")
            
            if discrepancy:
                print(f"  ⚠️  {discrepancy}")
            else:
                print(f"  ✓ Percentage totals match: YES")
            
            # Add all trades to master list
            all_data.extend(live_trades)
            all_data.extend(demo_trades)
            
        except Exception as e:
            print(f"  ❌ Error processing sheet {sheet_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_data, sheet_processing_summary, len(monthly_sheets)

def parse_demo_date(date_str, sheet_name):
    """Parse demo date string and convert to proper format"""
    if pd.isna(date_str) or str(date_str).strip() in ['', 'nan', 'NaN']:
        return ''
    
    date_str = str(date_str).strip()
    
    # Try common date formats first
    date_formats = [
        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y',
        '%d %b %Y', '%d %B %Y', '%b %d, %Y', '%B %d, %Y'
    ]
    
    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    # Try to extract date from strings like "Mon 08/01/2024"
    date_patterns = [
        r'(\d{1,2}/\d{1,2}/\d{2,4})',  # dd/mm/yyyy or mm/dd/yyyy
        r'(\d{1,2}-\d{1,2}-\d{2,4})',  # dd-mm-yyyy or mm-dd-yyyy
        r'(\d{4}-\d{1,2}-\d{1,2})',  # yyyy-mm-dd
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, date_str)
        if match:
            date_part = match.group(1)
            # Try different date formats
            for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d']:
                try:
                    dt = datetime.strptime(date_part, fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue
    
    # If it's just a day name, use the sheet name for context
    if any(day in date_str.lower() for day in ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']):
        # Extract month and year from sheet name
        month_map = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
            'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
        }
        
        # Find year in sheet name
        year_match = re.search(r'(\d{2,4})', sheet_name)
        if year_match:
            year_str = year_match.group(1)
            year = f"20{year_str}" if len(year_str) == 2 else year_str
            
            # Find month in sheet name
            for month_name, month_num in month_map.items():
                if month_name in sheet_name.lower():
                    # Map day names to approximate dates
                    day_map = {
                        'mon': '01', 'tue': '02', 'wed': '03', 'thu': '04',
                        'fri': '05', 'sat': '06', 'sun': '07'
                    }
                    
                    for day_name, day_num in day_map.items():
                        if day_name in date_str.lower():
                            return f"{year}-{month_num}-{day_num}"
                    
                    # Default to 1st of month
                    return f"{year}-{month_num}-01"
    
    return date_str

def clean_and_reorder_row(row, sheet_name=None):
    """Clean individual trade row and ensure correct column order"""
    # Ensure row has at least 9 elements
    if len(row) < 9:
        row = list(row) + [''] * (9 - len(row))
    
    # Map columns based on typical Excel structure
    # Expected order in Excel: DATE, TIME, PAIR, TRADE IMAGE, TYPE, RESULT, GROWTH %, Comments
    # We need: DATE, TIME, LIVEorDEMO, PAIR, TYPE, RESULT, GROWTH %, TRADE IMAGE, Comments
    
    # Extract values with defaults
    date_val = row[0] if len(row) > 0 else ''
    time_val = row[1] if len(row) > 1 else ''
    pair_val = row[2] if len(row) > 2 else ''
    trade_image_val = row[3] if len(row) > 3 else ''
    type_val = row[4] if len(row) > 4 else ''
    result_val = row[5] if len(row) > 5 else ''
    growth_val = row[6] if len(row) > 6 else ''
    comments_val = row[7] if len(row) > 7 else ''
    status_val = row[8] if len(row) > 8 else ''
    
    # Clean date
    if pd.isna(date_val) or str(date_val).strip() in ['', 'nan', 'NaN']:
        clean_date = ''
    else:
        date_str = str(date_val).strip()
        
        # Handle Excel serial dates
        if isinstance(date_val, (int, float)) and 40000 < date_val < 50000:
            try:
                dt = pd.Timestamp('1899-12-30') + pd.Timedelta(days=float(date_val))
                clean_date = dt.strftime('%Y-%m-%d')
            except:
                clean_date = date_str
        else:
            # Parse date string
            if status_val == 'DEMO' and sheet_name:
                clean_date = parse_demo_date(date_str, sheet_name)
            else:
                # Try standard formats
                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y']:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        clean_date = dt.strftime('%Y-%m-%d')
                        break
                    except ValueError:
                        clean_date = date_str
    
    # Clean other values
    clean_time = '' if pd.isna(time_val) else str(time_val).strip()
    clean_pair = '' if pd.isna(pair_val) else str(pair_val).strip()
    clean_trade_image = '' if pd.isna(trade_image_val) else str(trade_image_val).strip()
    clean_type = '' if pd.isna(type_val) else str(type_val).strip()
    clean_result = '' if pd.isna(result_val) else str(result_val).strip().upper()
    clean_comments = '' if pd.isna(comments_val) else str(comments_val).strip()
    
    # Remove trader names from comments
    clean_comments = re.sub(r'\b(David|Steve|DAVID|STEVE)\b', '', clean_comments, flags=re.IGNORECASE)
    clean_comments = re.sub(r'\s+', ' ', clean_comments).strip()
    
    # Clean growth %
    if pd.isna(growth_val) or str(growth_val).strip() in ['', 'nan', 'NaN']:
        clean_growth = ''
    else:
        try:
            growth_num = float(growth_val)
            clean_growth = f"{growth_num:.4f}".rstrip('0').rstrip('.')
            if clean_growth == '':
                clean_growth = '0'
        except (ValueError, TypeError):
            clean_growth = str(growth_val).strip()
    
    # Clean status
    if pd.isna(status_val) or str(status_val).strip() in ['', 'nan', 'NaN']:
        clean_status = 'LIVE'
    else:
        status_str = str(status_val).strip().upper()
        clean_status = 'LIVE' if 'LIVE' in status_str else ('DEMO' if 'DEMO' in status_str else 'LIVE')
    
    # Return in correct order: DATE, TIME, LIVEorDEMO, PAIR, TYPE, RESULT, GROWTH %, TRADE IMAGE, Comments
    return [
        clean_date,
        clean_time,
        clean_status,
        clean_pair,
        clean_type,
        clean_result,
        clean_growth,
        clean_trade_image,
        clean_comments
    ]

def create_final_dataframe(raw_data, sheet_summary):
    """Create final DataFrame with proper column structure and ordering"""
    
    # Correct column order for final output
    columns = ['DATE', 'TIME', 'LIVEorDEMO', 'PAIR', 'TYPE', 'RESULT', 'GROWTH %', 'TRADE IMAGE', 'Comments']
    
    # Process each row with sheet context
    valid_data = []
    
    # Track current sheet for date parsing
    current_sheet_index = 0
    rows_processed = 0
    
    for row in raw_data:
        # Determine which sheet this row came from
        sheet_name = None
        if current_sheet_index < len(sheet_summary):
            current_sheet = sheet_summary[current_sheet_index]
            total_rows = current_sheet['Live_Records'] + current_sheet['Demo_Records']
            
            if rows_processed < total_rows:
                sheet_name = current_sheet['Sheet_Name']
                rows_processed += 1
            else:
                # Move to next sheet
                current_sheet_index += 1
                rows_processed = 0
                if current_sheet_index < len(sheet_summary):
                    current_sheet = sheet_summary[current_sheet_index]
                    sheet_name = current_sheet['Sheet_Name']
                    rows_processed = 1
        
        # Clean and reorder the row
        cleaned_row = clean_and_reorder_row(row, sheet_name)
        
        # Check if this is a valid trade row (has a date)
        if cleaned_row[0] and cleaned_row[0].strip():
            valid_data.append(cleaned_row)
    
    # Create DataFrame
    if not valid_data:
        print("  ⚠️ Warning: No valid data found")
        return pd.DataFrame(columns=columns)
    
    df = pd.DataFrame(valid_data, columns=columns)
    
    # Remove any rows with empty dates
    df = df[df['DATE'] != '']
    
    return df

def print_processing_summary(sheet_summary, total_monthly_sheets):
    """Print detailed processing summary"""
    
    print(f"\n{'='*100}")
    print(f"PROCESSING SUMMARY ({total_monthly_sheets} Monthly Sheets from Jan 2024 to Jan 2026)")
    print(f"{'='*100}")
    
    print(f"\n{'Sheet Name':<12} {'Live':<6} {'Demo':<6} {'Source Total %':<15} {'Calculated Total %':<18} {'Match':<8} {'Discrepancy':<15}")
    print(f"{'-'*100}")
    
    total_live = 0
    total_demo = 0
    sheets_with_discrepancies = 0
    
    for summary in sheet_summary:
        sheet_name = summary['Sheet_Name']
        live = summary['Live_Records']
        demo = summary['Demo_Records']
        source_pct = f"{summary['Source_Total_Percentage']:.4f}" if summary['Source_Total_Percentage'] is not None else "N/A"
        calc_pct = f"{summary['Calculated_Total_Percentage']:.4f}"
        
        # Check if totals match
        match = "YES"
        if summary['Discrepancy']:
            match = "NO"
            sheets_with_discrepancies += 1
        
        discrepancy = f"{summary['Discrepancy_Amount']:.4f}" if summary['Discrepancy_Amount'] is not None else "0.0000"
        
        print(f"{sheet_name:<12} {live:<6} {demo:<6} {source_pct:<15} {calc_pct:<18} {match:<8} {discrepancy:<15}")
        
        total_live += live
        total_demo += demo
    
    print(f"{'-'*100}")
    print(f"{'TOTALS':<12} {total_live:<6} {total_demo:<6} {'':<15} {'':<18} {'':<8} {'':<15}")
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total monthly sheets processed: {len(sheet_summary)}/{total_monthly_sheets}")
    print(f"   Total LIVE records: {total_live}")
    print(f"   Total DEMO records: {total_demo}")
    print(f"   Total all records: {total_live + total_demo}")
    print(f"   Sheets with discrepancies: {sheets_with_discrepancies}")
    
    if sheets_with_discrepancies == 0:
        print(f"   ✅ All sheets processed successfully with matching totals!")
    else:
        print(f"   ⚠️  {sheets_with_discrepancies} sheet(s) have percentage discrepancies")

def main():
    # File path
    file_path = "raw_dataset.xlsx"
    
    print(f"{'='*100}")
    print("TRADE JOURNAL EXTRACTION TOOL")
    print("Extracting data from 25 monthly sheets (Jan 2024 to Jan 2026)")
    print(f"{'='*100}")
    
    # Extract data from all sheets
    raw_data, sheet_summary, total_monthly_sheets = extract_trade_data(file_path)
    
    # Create final DataFrame
    final_df = create_final_dataframe(raw_data, sheet_summary)
    
    # Print processing summary
    print_processing_summary(sheet_summary, total_monthly_sheets)
    
    # Print sample
    print(f"\n{'='*100}")
    print("SAMPLE OF EXTRACTED DATA (First 5 rows)")
    print(f"{'='*100}")
    print(final_df.head().to_string(index=False))
    
    # Save to CSV
    output_file = "trade_journal_corrected.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*100}")
    print("EXTRACTION COMPLETE!")
    print(f"{'='*100}")
    print(f"📁 CSV saved as: {output_file}")
    print(f"📊 Total records extracted: {len(final_df)}")
    print(f"   • LIVE records: {len(final_df[final_df['LIVEorDEMO'] == 'LIVE'])}")
    print(f"   • DEMO records: {len(final_df[final_df['LIVEorDEMO'] == 'DEMO'])}")
    
    return final_df

if __name__ == "__main__":
    df = main()