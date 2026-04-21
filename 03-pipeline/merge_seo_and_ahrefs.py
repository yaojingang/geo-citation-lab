import pandas as pd
import numpy as np

def merge_align_and_save(seo_path, ahrefs_path, output_filename='merged_seo_data.csv'):
    # 1. 读取数据
    df_seo = pd.read_csv(seo_path)
    df_ahrefs = pd.read_csv(ahrefs_path)
    
    # 2. 统一公共列名（去除首尾空格）
    df_seo['引用域名'] = df_seo['引用域名'].str.strip()
    df_ahrefs['引用域名'] = df_ahrefs['引用域名'].str.strip()
    
    # 3. 量纲对齐：将 Ahrefs 的 1-100 乘以 10
    df_ahrefs['DR_Scaled'] = pd.to_numeric(df_ahrefs['DR'], errors='coerce') * 10 
    
    # 4. 左连接：以 DataForSEO 为基准
    df_merged = pd.merge(
        df_seo, 
        df_ahrefs, 
        on=['文件名', '引用域名'], 
        how='left',
        suffixes=('_seo', '_ahrefs')
    )
    
    # 5. 数据修补与 Final_DR 计算
    seo_dr_numeric = pd.to_numeric(df_merged['域名评级(Domain Rank)'], errors='coerce')
    df_merged['最终评级(Final_DR)'] = seo_dr_numeric.fillna(df_merged['DR_Scaled'])
    
    # 标记数据来源，方便追溯
    df_merged['评级数据来源'] = np.where(seo_dr_numeric.notna(), 'DataForSEO', 
                                   np.where(df_merged['DR_Scaled'].notna(), 'Ahrefs_Converted', '无数据'))
    
    # 统一网站类型
    if '网站类型_seo' in df_merged.columns:
        df_merged['网站类型'] = df_merged['网站类型_seo'].fillna(df_merged.get('网站类型_ahrefs'))
    
    # 6. 清理冗余列
    cols_to_remove = ['网站类型_seo', '网站类型_ahrefs', 'DR_Scaled', 'DR']
    df_final = df_merged.drop(columns=[c for c in cols_to_remove if c in df_merged.columns])

    # ==========================================
    # 【新增】保存为 CSV 文件
    # ==========================================
    # encoding='utf_8_sig' 可以确保在 Windows Excel 中打开时中文不乱码
    df_final.to_csv(output_filename, index=False, encoding='utf_8_sig')
    
    print(f"✅ 处理完成！已成功保存至: {output_filename}")
    print(f"📊 最终 DR 来源统计:\n{df_final['评级数据来源'].value_counts()}")
    
    return df_final

# ------------------------------------------
# 使用方法：
# ------------------------------------------
result_df = merge_align_and_save('google _analysis_results.csv', 'results_google_ahrefs_A_commerce.csv', 'A.csv')