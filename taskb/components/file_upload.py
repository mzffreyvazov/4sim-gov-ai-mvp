import streamlit as st
import pandas as pd
from typing import Optional

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
