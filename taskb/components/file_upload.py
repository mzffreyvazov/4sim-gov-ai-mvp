import streamlit as st
import pandas as pd
from typing import Optional, Tuple

def enhanced_file_uploader() -> Optional[st.runtime.uploaded_file_manager.UploadedFile]:
    """Enhanced file uploader with validation and preview"""
    
    st.subheader("📁 Upload Your Dataset")
    st.markdown("Upload a CSV or Excel file to get started with AI-powered dashboard generation.")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Supported formats: CSV, Excel (.xlsx, .xls). Maximum file size: 200MB"
    )
    
    if uploaded_file is not None:
        # File info
        file_details = {
            "Filename": uploaded_file.name,
            "File type": uploaded_file.type,
            "File size": f"{uploaded_file.size / 1024 / 1024:.2f} MB"
        }
        
        with st.expander("📋 File Information", expanded=True):
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
        
        # Validation
        if uploaded_file.size > 200 * 1024 * 1024:  # 200MB limit
            st.error("❌ File too large! Please upload a file smaller than 200MB.")
            return None
            
        st.success("✅ File uploaded successfully!")
        
    return uploaded_file

def display_dataframe_preview(df: pd.DataFrame):
    """Display simple DataFrame preview - first 20 rows as-is"""
    
    st.subheader("📊 Dataset Preview")
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    with col4:
        missing_values = df.isnull().sum().sum()
        st.metric("Missing Values", missing_values)
    
    # Display first 20 rows
    st.markdown("**First 20 rows of your dataset:**")
    
    try:
        # Display the DataFrame as-is without any cleaning
        st.dataframe(df.head(20), use_container_width=True)
    except Exception:
        # Fallback to table if dataframe fails
        st.table(df.head(20))
    
    # Column information in expander
    with st.expander("📋 Column Information"):
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.count(),
            "Null Count": df.isnull().sum(),
            "Unique Values": [df[col].nunique() for col in df.columns]
        })
        
        try:
            st.dataframe(col_info, use_container_width=True)
        except Exception:
            st.table(col_info)

def csv_formatting_component(df: pd.DataFrame, raw_csv_data: str, api_key: str, filename: str) -> Tuple[Optional[pd.DataFrame], bool]:
    """
    Component for AI-powered CSV formatting with user control
    
    Args:
        df: Current DataFrame
        raw_csv_data: Raw CSV data as string
        api_key: API key for AI processing
        filename: Original filename
        
    Returns:
        Tuple of (formatted_dataframe, was_formatted)
    """
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.parent
    sys.path.append(str(current_dir))
    
    from utils.ai_agents import CSVFormatter
    from utils.data_processing import DataProcessor
    
    st.subheader("🤖 AI-Powered CSV Formatting")
    
    # Check if formatting is recommended
    needs_formatting = DataProcessor.detect_if_needs_formatting(df)
    
    if needs_formatting:
        st.warning("⚠️ **Data Formatting Recommended**")
        st.markdown("""
        The uploaded file appears to have complex structure that might benefit from AI formatting:
        - Multi-level headers
        - Unstructured column names  
        - Mixed data types in headers
        
        Our AI can automatically clean and structure this data for better analysis.
        """)
    else:
        st.info("ℹ️ **Optional Formatting Available**")
        st.markdown("""
        Your data appears well-structured, but you can still use AI formatting to:
        - Standardize column names
        - Clean inconsistent formatting
        - Optimize for analysis
        """)
    
    # Show preview of current data issues (if any)
    col1, col2 = st.columns(2)
    
    with col1:
        if needs_formatting:
            st.markdown("**Current Issues Detected:**")
            issues = []
            
            # Check for unnamed columns
            unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
            if unnamed_cols:
                issues.append(f"• {len(unnamed_cols)} unnamed columns")
            
            # Check for duplicate columns
            if len(df.columns) != len(set(df.columns)):
                issues.append("• Duplicate column names")
            
            # Check for very long column names
            long_cols = [col for col in df.columns if len(str(col)) > 30]
            if long_cols:
                issues.append(f"• {len(long_cols)} overly long column names")
            
            for issue in issues[:5]:  # Show max 5 issues
                st.write(issue)
    
    with col2:
        st.markdown("**AI Formatting Benefits:**")
        st.write("• Flatten multi-level headers")
        st.write("• Standardize column names")
        st.write("• Handle missing values properly")
        st.write("• Make data analysis-ready")
        st.write("• Preserve original data integrity")
    
    # Formatting options - simplified to single button
    col1, col2 = st.columns([2, 3])
    
    with col1:
        format_data = st.button(
            "🤖 Format with AI",
            type="primary" if needs_formatting else "secondary",
            help="Use AI to automatically format, preview, and save the clean CSV data",
            use_container_width=True
        )
    
    with col2:
        if needs_formatting:
            st.info("💡 **Recommended**: This data would benefit from AI formatting")
        else:
            st.success("✅ **Optional**: Your data looks well-structured already")
    
    # Handle user action
    if format_data:
        if not api_key:
            st.error("❌ Please provide an API key in the sidebar to use AI formatting.")
            return df, False
        
        return _perform_csv_formatting_with_auto_save(raw_csv_data, api_key, filename)
    
    # If no action taken, return original
    return df, False

def _perform_csv_formatting_with_auto_save(raw_csv_data: str, api_key: str, filename: str) -> Tuple[Optional[pd.DataFrame], bool]:
    """Perform CSV formatting with automatic preview and save"""
    try:
        # Import here to avoid circular imports
        import sys
        from pathlib import Path
        current_dir = Path(__file__).parent.parent
        sys.path.append(str(current_dir))
        
        from utils.ai_agents import CSVFormatter
        from utils.data_processing import DataProcessor
        
        with st.spinner("🤖 AI is analyzing and formatting your data..."):
            # Initialize CSV formatter
            formatter = CSVFormatter(api_key=api_key)
            
            # Format the CSV data
            formatted_csv = formatter.format_csv_data(raw_csv_data)
            
            if not formatted_csv.strip():
                st.error("❌ AI formatting failed - no valid data returned.")
                return None, False
            
            # Parse the formatted CSV
            formatted_df = DataProcessor.parse_formatted_csv_string(formatted_csv)
            
            # Show success message
            st.success("✅ **Data Successfully Formatted and Processed!**")
            
            # Show automatic preview
            with st.expander("📊 Formatted Data Preview", expanded=True):
                # Comparison metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Try to get original row count for comparison
                    try:
                        import io
                        original_df = pd.read_csv(io.StringIO(raw_csv_data))
                        original_rows = len(original_df)
                    except:
                        original_rows = "N/A"
                    
                    st.metric("Rows", f"{len(formatted_df)}", 
                             delta=f"vs {original_rows} original" if original_rows != "N/A" else None)
                
                with col2:
                    st.metric("Columns", len(formatted_df.columns))
                
                with col3:
                    clean_cols = [col for col in formatted_df.columns if not str(col).startswith('Unnamed:')]
                    st.metric("Clean Columns", len(clean_cols))
                
                # Show column name improvements
                st.markdown("**✨ Improved Column Names:**")
                col_names_preview = ", ".join(formatted_df.columns[:5])
                if len(formatted_df.columns) > 5:
                    col_names_preview += f", ... (+{len(formatted_df.columns)-5} more)"
                st.write(col_names_preview)
                
                # Show data preview
                st.markdown("**📄 Data Preview (first 10 rows):**")
                st.dataframe(formatted_df.head(10), use_container_width=True)
            
            # Show save confirmation
            st.info("💾 **Formatted data is ready for AI analysis!** The cleaned dataset will be used for generating chart suggestions.")
            
            return formatted_df, True
                
    except Exception as e:
        st.error(f"❌ Error during formatting: {str(e)}")
        return None, False
