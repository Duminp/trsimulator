import pandas as pd
import glob
import os
import re

def aggregate_journals():
    # 1. Identify all potential data sources starting with "EU_USD"
    all_potential_files = glob.glob("EU_USD*")
    
    # Filter out target files and calculation sheets
    target_files = [
        f for f in all_potential_files 
        if "tradeJournal.csv" not in f 
        and "Data Calcs" not in f 
        and not f.endswith(".tmp")
    ]
    
    if not target_files:
        print("No valid trading data files (CSV or XLSX) found in the directory.")
        return

    all_data = []
    verification_log = []
    
    # Mapping for month validation and forced context
    month_map = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
        'JUNE': 6, 'JULY': 7, 'SEPT': 9
    }

    print(f"Starting Context-Locked Aggregation of {len(target_files)} potential source(s)...\n")

    for file in target_files:
        try:
            current_sources = []
            
            # --- EXTRACT CONTEXT FROM FILENAME ---
            file_match = re.search(r'([A-Za-z]+)\s*(\d{2})', file)
            f_month_name = file_match.group(1).upper()[:3] if file_match else None
            f_year_short = file_match.group(2) if file_match else None
            
            exp_month = month_map.get(f_month_name)
            exp_year = int("20" + f_year_short) if f_year_short else None

            # --- CASE 1: Excel Workbook ---
            if file.lower().endswith(".xlsx"):
                # Use engine='openpyxl' to ensure compatibility
                workbook = pd.read_excel(file, sheet_name=None, engine='openpyxl')
                for sheet_name, df in workbook.items():
                    if any(x in sheet_name.upper() for x in ["DATA CALCS", "SUMMARY", "SHEET"]):
                        continue
                    
                    # Determine context for this specific sheet
                    s_match = re.search(r'([A-Za-z]+)\s*(\d{2})', sheet_name)
                    s_month = month_map.get(s_match.group(1).upper()[:3]) if s_match else exp_month
                    s_year = int("20" + s_match.group(2)) if s_match else exp_year
                    
                    current_sources.append((df, f"Sheet: {sheet_name}", s_month, s_year))

            # --- CASE 2: CSV Exports ---
            elif file.lower().endswith(".csv"):
                try:
                    # Individual CSVs often have different encodings
                    try:
                        df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
                    except:
                        df = pd.read_csv(file, encoding='latin1', on_bad_lines='skip')
                    
                    current_sources.append((df, f"CSV: {file}", exp_month, exp_year))
                except Exception as e:
                    print(f"  ⚠️ Skipping {file}: {e}")
                    continue

            # --- PROCESS DATA ---
            for df, source_label, ctx_month, ctx_year in current_sources:
                # Clean headers
                df.columns = [str(c).strip().upper() for c in df.columns]
                
                if 'DATE' in df.columns:
                    # Drop completely empty rows
                    df = df.dropna(subset=['DATE'], how='all')
                    raw_count = len(df)
                    
                    # Pre-clean numeric column
                    if 'GROWTH %' in df.columns:
                        df['GROWTH %'] = pd.to_numeric(df['GROWTH %'], errors='coerce').fillna(0)
                    else:
                        df['GROWTH %'] = 0

                    def clean_and_fix_date(val):
                        val = str(val).strip()
                        # Ignore header rows, blank entries, or summary box text
                        if not val or val.lower() == 'nan' or val.upper() == 'DATE' or 'TRADING' in val.upper(): 
                            return pd.NaT
                        
                        # Remove weekday prefixes (e.g. "Tue ", "Fri ")
                        clean_val = re.sub(r'^[A-Za-z]+\s+', '', val) 
                        
                        try:
                            # Primary parse using UK/Day-First format
                            dt = pd.to_datetime(clean_val, dayfirst=True, errors='coerce')
                            
                            if pd.notna(dt) and ctx_month and ctx_year:
                                # Logic to prevent parsing flips (e.g. 11/02 being Nov instead of Feb)
                                if dt.month != ctx_month:
                                    alt_dt = pd.to_datetime(clean_val, dayfirst=False, errors='coerce')
                                    if pd.notna(alt_dt) and alt_dt.month == ctx_month:
                                        dt = alt_dt
                                
                                # FORCE the context year and month from the filename/sheet
                                try:
                                    dt = dt.replace(year=ctx_year, month=ctx_month)
                                except ValueError:
                                    # Day invalid for this month (typo in sheet), attempt fallback to last day
                                    return pd.NaT
                            return dt
                        except:
                            return pd.NaT

                    df['DATE'] = df['DATE'].apply(clean_and_fix_date)
                    
                    # Filter: Keep any row that has a valid DATE
                    df = df[df['DATE'].notna()]
                    
                    if not df.empty:
                        # Corrected: Use .str accessor before calling .upper() on the Series
                        if 'RESULT' in df.columns:
                            df['RESULT'] = df['RESULT'].fillna('').astype(str).str.strip().str.upper()
                            # Standardize Result mappings
                            df.loc[df['RESULT'].str.contains('WIN|WON', na=False), 'RESULT'] = 'WON'
                            df.loc[df['RESULT'].str.contains('LOSS|LOST', na=False), 'RESULT'] = 'LOST'
                            # If result is empty but growth is 0, assume breakeven
                            df.loc[(df['RESULT'] == '') & (df['GROWTH %'] == 0), 'RESULT'] = 'BREAKEVEN'
                            # Ensure BE variants are caught
                            df.loc[df['RESULT'].str.contains('BE|BREAK', na=False), 'RESULT'] = 'BREAKEVEN'
                        else:
                            df['RESULT'] = 'UNKNOWN'

                        # Standardize Pair and Time as strings
                        for col in ['PAIR', 'TIME', 'TYPE', 'COMMENTS', 'TRADE IMAGE']:
                            if col in df.columns:
                                df[col] = df[col].fillna('').astype(str).str.strip()
                        
                        # Normalize Column Names for the final merge
                        mapping = {'IMAGE': 'TRADE IMAGE', 'TRADE IMAGE': 'TRADE IMAGE', 
                                   'COMMENTS ': 'COMMENTS', 'Comments': 'COMMENTS'}
                        df = df.rename(columns=mapping)
                        
                        cols_to_keep = ['DATE', 'TIME', 'PAIR', 'TYPE', 'RESULT', 'GROWTH %', 'COMMENTS', 'TRADE IMAGE']
                        existing = [c for c in cols_to_keep if c in df.columns]
                        extracted = df[existing].copy()
                        
                        all_data.append(extracted)
                        verification_log.append({
                            "Source": source_label,
                            "Rows in File": raw_count,
                            "Extracted Trades": len(extracted),
                            "Noise Rows Filtered": raw_count - len(extracted),
                            "Locked Period": f"{ctx_month}/{ctx_year}" if ctx_month else "N/A"
                        })

        except Exception as e:
            print(f"❌ Error in {file}: {e}")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Ensure all standard columns exist in the final master file
        for col in ['TIME', 'PAIR', 'TYPE', 'RESULT', 'GROWTH %', 'COMMENTS', 'TRADE IMAGE']:
            if col not in final_df.columns: 
                final_df[col] = "" if col != 'GROWTH %' else 0.0
        
        # Sort chronologically to ensure equity curves are accurate
        final_df = final_df.sort_values('DATE')
        final_df.to_csv("tradeJournal.csv", index=False)
        
        # --- AUDIT LOG ---
        print("\n" + "="*95)
        print(f"{'DATA AGGREGATION & AUDIT LOG':^95}")
        print("="*95)
        log_df = pd.DataFrame(verification_log)
        # Note: Noise Rows Filtered includes headers, summary boxes, and empty spacer rows
        print(log_df.to_string(index=False))
        print("="*95)
        print(f"SUCCESS: 'tradeJournal.csv' generated with {len(final_df)} records.")
        print(f"Verified Date Range: {final_df['DATE'].min().date()} to {final_df['DATE'].max().date()}")
        print("="*95)
    else:
        print("\nAggregation Failed: No valid data found in provided files.")

if __name__ == "__main__":
    aggregate_journals()