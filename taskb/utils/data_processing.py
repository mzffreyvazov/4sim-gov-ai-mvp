import pandas as pd
import chardet
import io
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
