import pandas as pd
import chardet
import io
import tempfile
import os
from typing import Dict, Tuple, Optional, List

class DataProcessor:
    """Handles file upload and DataFrame operations"""
    
    @staticmethod
    def read_uploaded_file(file_content: bytes, filename: str) -> pd.DataFrame:
        """Read uploaded file content into DataFrame with encoding detection"""
        try:
            if filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(file_content))
            else:  # Assume CSV
                encoding = chardet.detect(file_content)['encoding'] or 'utf-8'
                df = pd.read_csv(io.StringIO(file_content.decode(encoding, errors='ignore')))
            
            return df
            
        except Exception as e:
            raise ValueError(f"Could not parse the file. Error: {e}")
    
    @staticmethod
    def get_dataframe_context(df: pd.DataFrame) -> Dict[str, str]:
        """Generate comprehensive DataFrame context for AI analysis"""
        df_head_str = df.head(20).to_string()
        df_shape_str = str(df.shape)
        df_columns_str = str(df.columns.tolist())
        df_description_str = df.describe(include='all').to_string()
        
        with io.StringIO() as buf:
            df.info(buf=buf)
            df_info_str = buf.getvalue()

        return {
            "df_head": df_head_str,
            "df_shape": df_shape_str,
            "df_columns": df_columns_str,
            "df_description": df_description_str,
            "df_info": df_info_str,
        }
    
    @staticmethod
    def validate_columns(df: pd.DataFrame, column_mapping: Dict[str, Optional[str]]) -> Tuple[bool, List[str]]:
        """Validate that required columns exist in DataFrame"""
        required_cols = set()
        for key, value in column_mapping.items():
            if value and isinstance(value, str) and value.strip():
                clean_value = value.strip()
                if clean_value in df.columns:
                    required_cols.add(clean_value)
        
        missing_cols = required_cols - set(df.columns)
        return len(missing_cols) == 0, list(missing_cols)
    
    @staticmethod
    def extract_raw_csv_data(file_content: bytes, filename: str) -> str:
        """Extract raw CSV data as string for AI formatting"""
        try:
            if filename.endswith((".xlsx", ".xls")):
                # For Excel files, convert to CSV string
                df = pd.read_excel(io.BytesIO(file_content))
                return df.to_csv(index=False)
            else:
                # For CSV files, return raw content as string
                encoding = chardet.detect(file_content)['encoding'] or 'utf-8'
                return file_content.decode(encoding, errors='ignore')
        except Exception as e:
            raise ValueError(f"Could not extract raw data from file. Error: {e}")
    
    @staticmethod
    def save_formatted_csv(formatted_csv_data: str, original_filename: str) -> Tuple[str, pd.DataFrame]:
        """
        Save formatted CSV data to a temporary file and return path and DataFrame
        
        Args:
            formatted_csv_data: The formatted CSV data as string
            original_filename: Original filename for reference
            
        Returns:
            Tuple of (temp_file_path, DataFrame)
        """
        try:
            # Create temporary file
            temp_dir = tempfile.gettempdir()
            base_name = os.path.splitext(original_filename)[0]
            temp_filename = f"{base_name}_formatted.csv"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            # Save formatted CSV
            with open(temp_path, 'w', encoding='utf-8', newline='') as f:
                f.write(formatted_csv_data)
            
            # Load as DataFrame to validate
            df = pd.read_csv(temp_path)
            
            return temp_path, df
            
        except Exception as e:
            raise ValueError(f"Could not save formatted CSV data. Error: {e}")
    
    @staticmethod
    def parse_formatted_csv_string(formatted_csv_data: str) -> pd.DataFrame:
        """Parse formatted CSV string directly into DataFrame"""
        try:
            return pd.read_csv(io.StringIO(formatted_csv_data))
        except Exception as e:
            raise ValueError(f"Could not parse formatted CSV data. Error: {e}")
    
    @staticmethod
    def detect_if_needs_formatting(df: pd.DataFrame) -> bool:
        """
        Detect if a DataFrame likely needs CSV formatting
        Returns True if formatting is recommended
        """
        # Check for common indicators of unformatted data
        indicators = []
        
        # 1. Check for multi-level headers (columns with unnamed patterns)
        unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
        if len(unnamed_cols) > 0:
            indicators.append("unnamed_columns")
        
        # 2. Check for sparse data in first few rows (likely header rows)
        if len(df) > 3:
            first_rows_empty_ratio = df.head(3).isnull().sum().sum() / (3 * len(df.columns))
            if first_rows_empty_ratio > 0.3:  # More than 30% empty in first 3 rows
                indicators.append("sparse_header_rows")
        
        # 3. Check for numeric data in column names (year indicators)
        numeric_pattern_in_cols = any(any(char.isdigit() for char in str(col)) for col in df.columns)
        if numeric_pattern_in_cols:
            indicators.append("numeric_in_headers")
        
        # 4. Check for very long column names (might be concatenated)
        long_columns = [col for col in df.columns if len(str(col)) > 50]
        if len(long_columns) > 0:
            indicators.append("long_column_names")
        
        # 5. Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            indicators.append("duplicate_columns")
        
        # Return True if 2 or more indicators are present
        return len(indicators) >= 2
