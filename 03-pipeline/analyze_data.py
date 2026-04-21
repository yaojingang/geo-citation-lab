import pandas as pd
import re

import pandas as pd

# 1. 读取数据 (请将 'your_data.csv' 替换为你的实际文件路径)
# df = pd.read_csv('final_resutls/chatgpt_results.csv')

# 为了演示，这里假设 df 已经是读取好的 DataFrame
def analyze_seo_data(df):
    print("="*40)
    print("数据分析报告")
    print("="*40)
    
    # ---------------------------------------------------------
    # 任务 1: 分析在A开头的6个种类中，每个种类各引用了多少条网址
    # ---------------------------------------------------------
    # 使用正则表达式提取不包含数字的分类名 (例如从 A_commerce42 提取出 A_commerce)
    df['项目分类'] = df['文件名'].str.extract(r'(A_?[a-zA-Z]+)')
    
    # 过滤出符合 A_commerce, A_technology 等分类的数据
    valid_categories = ['A_commerce', 'A_technology', 'A_local', 'A_news', 'A_healthcare', 'A_finance']
    df_filtered = df[df['项目分类'].isin(valid_categories)]
    
    # 统计每个种类的引用网址数量
    category_counts = df_filtered.groupby('项目分类')['引用域名'].count()
    
    print("\n1. 各大类项目的网址引用总数:")
    for category in valid_categories:
        # 使用 .get() 避免某些分类在数据中完全没有而报错
        count = category_counts.get(category, 0)
        print(f"   - {category}: {count} 条")

    # ---------------------------------------------------------
    # 任务 2: 引用网址最多的一条问答（即单个“文件名”）引用了多少条
    # ---------------------------------------------------------
    # 按照文件名分组，计算每个文件名包含的域名数量
    refs_per_qa = df.groupby('文件名')['引用域名'].count()
    
    max_refs_count = refs_per_qa.max()
    max_refs_qa_name = refs_per_qa.idxmax() # 找出对应的是哪个文件名
    
    print("\n2. 单个问答(文件名)引用网址的最大数量:")
    print(f"   - 最多引用了 {max_refs_count} 条网址")
    print(f"   - 对应的问答/文件名是: {max_refs_qa_name}")

    # ---------------------------------------------------------
    # 任务 3: 重复引用最多的网址是哪个，引用了多少条（前20名）
    # ---------------------------------------------------------
    # 统计 '引用域名' 列的出现频次，并取前20名
    top_20_domains = df['引用域名'].value_counts().head(20)
    
    print("\n3. 重复引用最多的前20个网址:")
    for rank, (domain, count) in enumerate(top_20_domains.items(), 1):
        print(f"   Top {rank}: {domain} (共引用 {count} 次)")

# 如果在本地运行，取消下面的注释并提供正确的文件名
df = pd.read_csv('final_resutls/perplexity_results.csv')
# analyze_seo_data(df)

def analyze_advanced_seo_data(df):
    print("="*50)
    print("深度数据分析报告 (5大进阶维度)")
    print("="*50)

    # 预处理：提取项目大类 (A_commerce, A_technology等)
    # 如果上一段代码没有运行，这里做个安全兜底
    if '项目分类' not in df.columns:
        df['项目分类'] = df['文件名'].str.extract(r'(A_?[a-zA-Z]+)')
    
    valid_categories = ['A_commerce', 'A_technology', 'A_local', 'A_news', 'A_healthcare', 'A_finance']
    df_filtered = df[df['项目分类'].isin(valid_categories)].copy()

    # ---------------------------------------------------------
    # 1. 网站类型 (Website Type) 偏好分析
    # ---------------------------------------------------------
    print("\n1. 网站类型分布:")
    type_counts = df_filtered['网站类型'].value_counts(dropna=False)
    for site_type, count in type_counts.items():
        print(f"   - {site_type}: {count} 次")

    # ---------------------------------------------------------
    # 2. 域名权威性 (Domain Rank) 质量评估
    # ---------------------------------------------------------
    print("\n2. 各大类项目的域名评级 (Domain Rank) 评估:")
    # 将 Domain Rank 强制转换为数值型，非数字内容（如 'null' 或缺失值）会变成 NaN
    # df_filtered['Domain Rank 数值'] = pd.to_numeric(df_filtered['域名评级(Domain Rank)'], errors='coerce')
    df_filtered['Domain Rank 数值'] = pd.to_numeric(df_filtered['最终评级(Final_DR)'], errors='coerce')
    
    # 按项目分类计算平均值、中位数和有效的数据量
    rank_stats = df_filtered.groupby('项目分类')['Domain Rank 数值'].agg(['mean', 'median', 'count']).round(2)
    rank_stats.columns = ['平均评级', '中位数评级', '有效数据量']
    print(rank_stats.to_string())

    # ---------------------------------------------------------
    # 3. DataForSEO 抓取成功率分析
    # ---------------------------------------------------------
    print("\n3. DataForSEO 抓取状态统计:")
    status_counts = df_filtered['DataForSEO_状态'].value_counts(dropna=False)
    for status, count in status_counts.items():
        print(f"   - {status}: {count} 次")

    # ---------------------------------------------------------
    # 4. 国际化与语言分布 (Country & Language)
    # ---------------------------------------------------------
    print("\n4. 国家与语言分布 (Top 5):")
    country_counts = df_filtered['国家(Country)'].value_counts().head(5)
    print("   [主要国家]")
    for country, count in country_counts.items():
        print(f"   - {country}: {count} 次")
        
    lang_counts = df_filtered['语言(Language)'].value_counts().head(5)
    print("\n   [主要语言]")
    for lang, count in lang_counts.items():
        print(f"   - {lang}: {count} 次")

    # ---------------------------------------------------------
    # 5. 网站技术栈 (Technologies) 词频分析
    # ---------------------------------------------------------
    print("\n5. 网站核心技术栈分布 (Top 10):")
    all_techs = []
    # 提取非空的技术栈数据
    tech_series = df_filtered['技术栈详情(Technologies)'].dropna()
    
    for row in tech_series:
        # 按 '|' 分割字符串
        tech_list = str(row).split('|')
        for t in tech_list:
            # 数据清洗：使用正则去除类似 '[null]' 的内容，并去除首尾空格
            clean_tech = re.sub(r'\[.*?\]', '', t).strip()
            # 过滤掉空的或者字符串为 'null' 的脏数据
            if clean_tech and clean_tech.lower() != 'null':
                all_techs.append(clean_tech)
                
    # 统计词频
    if all_techs:
        tech_counts = pd.Series(all_techs).value_counts().head(10)
        for tech, count in tech_counts.items():
            print(f"   - {tech}: {count} 次")
    else:
        print("   - 未提取到有效技术栈数据")

# ==========================================
# 运行方法：
# 将您的 CSV 文件路径替换进去即可运行
# ==========================================
# df = pd.read_csv('A.csv')
analyze_advanced_seo_data(df)

import pandas as pd
import numpy as np

def analyze_b_layer_styles(df):
    print("="*50)
    print("B层：提示词风格对照层 (Style Control) 分析报告")
    print("="*50)

    # ---------------------------------------------------------
    # 1. 数据预处理：识别语义组与提示风格
    # ---------------------------------------------------------
    # 过滤出 B 开头的数据
    df_b = df[df['文件名'].str.startswith('B', na=False)].copy()
    
    if df_b.empty:
        print("未检测到 B 层数据，请检查文件名。")
        return

    # 提取数字 ID (例如 'B54' -> 54)
    df_b['B_ID'] = df_b['文件名'].str.extract(r'B(\d+)').astype(float)
    
    # 划分语义组 (每 3 个为一组，1-3为第1组，4-6为第2组...52-54为第18组)
    df_b['语义组'] = np.ceil(df_b['B_ID'] / 3).astype(int)
    
    # 映射提示词风格 (假设逻辑: 余数 1 为 Natural, 2 为 SourceRequired, 0 为 Role)
    # 如果你的对应关系不同，请直接修改这里的字典映射
    def get_style(b_id):
        mod = b_id % 3
        if mod == 1: return '1_Natural (自然)'
        elif mod == 2: return '2_SourceRequired (要求来源)'
        else: return '3_Role (角色扮演)'
        
    df_b['提示风格'] = df_b['B_ID'].apply(get_style)

    # ---------------------------------------------------------
    # 2. 核心指标对比 1：引用数量的激进程度
    # ---------------------------------------------------------
    print("\n[对比 1] 哪种风格触发的搜索引用最丰富？")
    # 统计每种风格、每个文件名的引用条数
    citation_counts = df_b.groupby(['提示风格', '文件名'])['引用域名'].count().reset_index()
    # 计算每种风格的平均引用条数
    avg_citations = citation_counts.groupby('提示风格')['引用域名'].mean().round(2)
    
    for style, avg in avg_citations.items():
        print(f"   - {style}: 平均每个 Prompt 引用 {avg} 条网址")

    # ---------------------------------------------------------
    # 3. 核心指标对比 2：信息源权威性 (Domain Rank)
    # ---------------------------------------------------------
    print("\n[对比 2] 哪种风格倾向于引用更高质量/权威的网站？")
    if '最终评级(Final_DR)' in df_b.columns:
        dr_col = '最终评级(Final_DR)'
    else:
        # 如果你没跑上一个合并代码，这里做个兼容
        df_b['Domain Rank 数值'] = pd.to_numeric(df_b['域名评级(Domain Rank)'], errors='coerce')
        dr_col = 'Domain Rank 数值'

    dr_stats = df_b.groupby('提示风格')[dr_col].agg(['mean', 'median']).round(2)
    dr_stats.columns = ['平均 Domain Rank', '中位数 Domain Rank']
    print(dr_stats.to_string())
    print("   *注: 如果 Role 风格的 DR 显著变高，说明赋予专家人设会让模型更倾向于搜索顶尖权威网站。")

    # ---------------------------------------------------------
    # 4. 核心指标对比 3：网站类型的偏好转移
    # ---------------------------------------------------------
    print("\n[对比 3] 不同风格对【网站类型】的偏好差异 (Top 3):")
    # 透视表：查看不同风格下，各类网站类型的占比
    type_pivot = pd.crosstab(df_b['提示风格'], df_b['网站类型'], normalize='index') * 100
    
    for style in type_pivot.index:
        top3_types = type_pivot.loc[style].sort_values(ascending=False).head(3)
        print(f"   【{style}】主要引用来源:")
        for site_type, pct in top3_types.items():
            print(f"      - {site_type}: {pct:.1f}%")

    # ---------------------------------------------------------
    # 5. 核心指标对比 4：信息源的聚焦程度 (独立域名去重)
    # ---------------------------------------------------------
    print("\n[对比 4] 信息源广度 (独立域名数量):")
    unique_domains = df_b.groupby('提示风格')['引用域名'].nunique()
    for style, count in unique_domains.items():
         print(f"   - {style}: 共涉及 {count} 个不同的独立域名")
         
    return df_b

# ==========================================
# 运行方法：
# ==========================================
# 假设 df_final 是你上一步合并好 DR 数据的 DataFrame
# df = pd.read_csv('chatgpt_results.csv')
# df_b_analyzed = analyze_b_layer_styles(df)

import pandas as pd
import numpy as np

def analyze_c_layer_language(df):
    print("="*55)
    print("C层：语言子实验层 (Language Control) 深度分析报告")
    print("="*55)

    # ---------------------------------------------------------
    # 0. 数据预处理：识别语义组与语言版本
    # ---------------------------------------------------------
    df_c = df[df['文件名'].str.startswith('C', na=False)].copy()
    
    if df_c.empty:
        print("未检测到 C 层数据，请检查文件名是否以 'C' 开头。")
        return

    # 提取数字 ID (例如 'C1' -> 1)
    df_c['C_ID'] = df_c['文件名'].str.extract(r'C(\d+)').astype(float)
    
    # 划分语义组 (每 2 个为一组，1-2为第1组，3-4为第2组...71-72为第36组)
    df_c['语义组'] = np.ceil(df_c['C_ID'] / 2).astype(int)
    
    # 映射语言 (单数=英文，双数=中文)
    df_c['语言版本'] = np.where(df_c['C_ID'] % 2 != 0, 'EN (英文)', 'ZH (中文)')

    # 为了准确计算“触发率”，我们需要按“文件名”去重，得到 Prompt 级别的摘要
    # 假设如果完全没有触发搜索，CSV中该文件名的 '是否触发搜索' 标注为 '否' 或没有引用域名
    prompt_level_df = df_c.groupby(['文件名', '语义组', '语言版本']).agg({
        '是否触发搜索': 'first', # 获取该 Prompt 是否触发了搜索
        '引用域名': 'count'     # 计算该 Prompt 引用了多少条网址 (作为丰富度指标)
    }).reset_index()
    prompt_level_df.rename(columns={'引用域名': '引用数量'}, inplace=True)

    # ---------------------------------------------------------
    # 1. 搜索触发率差异 (Trigger Rate)
    # ---------------------------------------------------------
    print("\n[核心发现 1] 搜索触发率差异 (EN vs ZH):")
    # 计算总触发率
    trigger_rates = prompt_level_df[prompt_level_df['是否触发搜索'] == '是'].groupby('语言版本')['文件名'].count() / \
                    prompt_level_df.groupby('语言版本')['文件名'].count() * 100
    
    for lang, rate in trigger_rates.items():
         print(f"   - {lang} 总体搜索触发率: {rate:.1f}%")

    # 寻找“分歧组”：同一个问题，英文搜了中文没搜，或者中文搜了英文没搜
    # 透视表：行是语义组，列是语言，值是是否触发
    pivot_trigger = prompt_level_df.pivot(index='语义组', columns='语言版本', values='是否触发搜索')
    if 'EN (英文)' in pivot_trigger.columns and 'ZH (中文)' in pivot_trigger.columns:
        diff_groups = pivot_trigger[pivot_trigger['EN (英文)'] != pivot_trigger['ZH (中文)']]
        print(f"\n   -> 发现 {len(diff_groups)} 个语义组在不同语言下触发策略不一致:")
        for idx, row in diff_groups.iterrows():
            print(f"      * 语义组 {idx}: 英文触发=[{row['EN (英文)']}], 中文触发=[{row['ZH (中文)']}]")

    # ---------------------------------------------------------
    # 2. 搜索来源差异 (Source Origins)
    # ---------------------------------------------------------
    print("\n[核心发现 2] 搜索来源地域与网站类型差异:")
    # 我们只看成功抓取到的网址数据
    df_valid_urls = df_c[df_c['引用域名'].notna()]
    
    # 国家差异
    if '国家(Country)' in df_valid_urls.columns:
        print("\n   [Top 3 引用数据源国家]")
        country_crosstab = pd.crosstab(df_valid_urls['语言版本'], df_valid_urls['国家(Country)'], normalize='index') * 100
        for lang in country_crosstab.index:
            top_countries = country_crosstab.loc[lang].sort_values(ascending=False).head(3)
            print(f"   {lang}: {', '.join([f'{c} ({p:.1f}%)' for c, p in top_countries.items()])}")

    # 网站类型差异
    if '网站类型' in df_valid_urls.columns:
        print("\n   [Top 3 偏好的网站类型]")
        type_crosstab = pd.crosstab(df_valid_urls['语言版本'], df_valid_urls['网站类型'], normalize='index') * 100
        for lang in type_crosstab.index:
            top_types = type_crosstab.loc[lang].sort_values(ascending=False).head(3)
            print(f"   {lang}: {', '.join([f'{t} ({p:.1f}%)' for t, p in top_types.items()])}")

    # ---------------------------------------------------------
    # 3. 搜索质量差异 (Quality & Completeness)
    # ---------------------------------------------------------
    print("\n[核心发现 3] 搜索质量差异 (基于评级与引用丰度):")
    
    # 维度A：信息完整度 (引用数量)
    avg_citations = prompt_level_df.groupby('语言版本')['引用数量'].mean().round(2)
    print(f"   - 平均引用信息源数量:")
    for lang, avg in avg_citations.items():
        print(f"     * {lang}: 每次提问平均引用 {avg} 个网址")
        
    # 维度B：信息源权威度 (Domain Rank)
    dr_col = '最终评级(Final_DR)' if '最终评级(Final_DR)' in df_valid_urls.columns else '域名评级(Domain Rank)'
    df_valid_urls['DR_数值'] = pd.to_numeric(df_valid_urls[dr_col], errors='coerce')
    dr_stats = df_valid_urls.groupby('语言版本')['DR_数值'].agg(['mean', 'median']).round(2)
    
    print(f"\n   - 引用网站权威性 (Domain Rank):")
    for lang in dr_stats.index:
        print(f"     * {lang}: 平均DR为 {dr_stats.loc[lang, 'mean']}, 中位数DR为 {dr_stats.loc[lang, 'median']}")

    print("\n   *数据解读提示: 如果中文的平均引用数量少，且引用的 Domain Rank 显著低于英文，")
    print("   *可能暗示模型在中文语境下更偏向‘直接生成内部知识’(更少搜索)，或中文高质量检索源匮乏。")

    return df_c

# ==========================================
# 运行方法：
# ==========================================
# 假设 df_final 是你之前清洗合并好的包含 A/B/C 层的全量 DataFrame
# df_c_analyzed = analyze_c_layer_language(df)

import pandas as pd
import re

def find_missing_queries(df):
    print("="*50)
    print("未触发搜索/缺失数据 编号排查报告")
    print("="*50)

    # 1. 定义每个前缀预期的最大编号
    # 如果您的 B 层和 C 层实际跑了 72 条，请在这里把 60 改成 72
    expected_max_ids = {
        'A_commerce': 72,
        'Atechnology': 72,
        'A_local': 72,
        'Anews': 72,
        'A_healthcare': 72,
        'A_finance': 72,
        'B': 60,  
        'C': 60,
        'D': 50
    }

    # 2. 从“文件名”列中拆分出“前缀”和“编号”
    # 例如：'A_commerce42' -> 前缀: 'A_commerce', 编号: 42
    # 使用正则表达式匹配：字母/下划线作为前缀，末尾的数字作为编号
    extracted = df['文件名'].str.extract(r'^([a-zA-Z_]+)(\d+)$')
    df['Prefix'] = extracted[0]
    df['Num_ID'] = pd.to_numeric(extracted[1], errors='coerce')

    # 去除解析失败的空行
    df_valid = df.dropna(subset=['Prefix', 'Num_ID']).copy()

    # 3. 逐个分类对比并找出缺失项
    total_missing = 0
    
    for prefix, max_id in expected_max_ids.items():
        # 获取该前缀下，实际存在的所有去重编号
        actual_ids = set(df_valid[df_valid['Prefix'] == prefix]['Num_ID'].astype(int))
        
        # 生成该前缀应有的所有编号集合 (1 到 max_id)
        expected_ids = set(range(1, max_id + 1))
        
        # 计算差集 (预期有 - 实际有 = 缺失的)
        missing_ids = expected_ids - actual_ids
        
        # 排序以便于阅读
        missing_ids_sorted = sorted(list(missing_ids))
        
        # 打印结果
        if missing_ids_sorted:
            print(f"\n[{prefix}] 类别 (预期 1-{max_id}) - 缺失 {len(missing_ids_sorted)} 项:")
            # 将列表转为逗号分隔的字符串，方便直接复制阅读
            missing_str = ", ".join(map(str, missing_ids_sorted))
            print(f"   -> 缺失编号: {missing_str}")
            total_missing += len(missing_ids_sorted)
        else:
            print(f"\n[{prefix}] 类别 (预期 1-{max_id}) - 数据完整，无缺失！")

    print("="*50)
    print(f"排查完毕，总共发现 {total_missing} 个可能未触发搜索的提问。")
    print("="*50)

# ==========================================
# 运行方法：
# 将您合并好的 DataFrame 传给这个函数即可
# ==========================================
# df_final = pd.read_csv('your_merged_data.csv')
# find_missing_queries(df)

import pandas as pd

def find_stubborn_urls(seo_path, ahrefs_path, output_file='Missing_In_Both.csv'):
    print("="*50)
    print("🕵️ 双端缺失数据排查 (DataForSEO & Ahrefs)")
    print("="*50)

    # 1. 读取两份原始数据
    df_seo = pd.read_csv(seo_path)
    df_ahrefs = pd.read_csv(ahrefs_path)

    # 2. 统一域名格式 (去除首尾空格并转小写，防止因为大小写或空格导致匹配失败)
    df_seo['干净域名'] = df_seo['引用域名'].str.strip().str.lower()
    df_ahrefs['干净域名'] = df_ahrefs['引用域名'].str.strip().str.lower()

    # 3. 找到 DataForSEO 抓取失败的集合
    # 假设失败的标志是 'DataForSEO_状态' 为 '无数据' 或 '网络异常'，或者评级为空
    seo_failed = df_seo[
        (df_seo['DataForSEO_状态'] != '成功') 
    ].copy()
    
    print(f"1. DataForSEO 未成功抓取的总数: {len(seo_failed)} 条")

    # 4. 获取 Ahrefs 里所有成功返回的域名集合
    ahrefs_success_urls = set(df_ahrefs['干净域名'].dropna())
    print(f"2. Ahrefs 成功返回的域名总数: {len(ahrefs_success_urls)} 个")

    # 5. 【核心逻辑】在 DataForSEO 失败的名单里，排除掉 Ahrefs 成功抓到的
    # 剩下的就是两边都没查到的
    stubborn_data = seo_failed[~seo_failed['干净域名'].isin(ahrefs_success_urls)].copy()

    # 清理辅助列
    stubborn_data.drop(columns=['干净域名'], inplace=True)

    print(f"\n🚨 排查结果: 发现 {len(stubborn_data)} 条双端均无数据的“顽固网址”！")

    # 6. 展示和保存结果
    if len(stubborn_data) > 0:
        print("\n[缺失名单预览 (前5条)]:")
        display_cols = ['文件名', '引用域名', 'DataForSEO_状态']
        # 防止有的列不存在报错
        display_cols = [c for c in display_cols if c in stubborn_data.columns]
        print(stubborn_data[display_cols].head().to_string(index=False))

        # 保存为 CSV 供你手动核查
        stubborn_data.to_csv(output_file, index=False, encoding='utf_8_sig')
        print(f"\n✅ 完整缺失名单已保存至: {output_file}")
    else:
        print("\n🎉 太棒了！所有 DataForSEO 缺失的数据，Ahrefs 都成功补全了！")

    return stubborn_data

# ==========================================
# 运行方法：替换为你实际的原始文件名
# ==========================================
# df_missing = find_stubborn_urls('chatgpt/results.csv', 'chatgpt/results_ahrefs.csv')

import pandas as pd
import numpy as np

def analyze_d_layer_edge_cases(df):
    print("="*55)
    print("D层：极端场景与边界测试 (Edge Cases) 深度分析报告")
    print("="*55)

    # ---------------------------------------------------------
    # 0. 数据预处理与场景分类
    # ---------------------------------------------------------
    df_d = df[df['文件名'].str.startswith('D', na=False)].copy()
    
    if df_d.empty:
        print("未检测到 D 层数据，请检查文件名是否以 'D' 开头。")
        return

    # 提取数字 ID (例如 'D42' -> 42)
    df_d['D_ID'] = df_d['文件名'].str.extract(r'D(\d+)').astype(float)
    
    # 映射 D 层的 5 大业务场景
    def map_d_scenario(d_id):
        if 1 <= d_id <= 10:
            return '1_高风险 (医疗/金融/法律)'
        elif 11 <= d_id <= 20:
            return '2_极度模糊 (发散/无明确实体)'
        elif 21 <= d_id <= 30:
            return '3_多约束复杂任务 (规划/对比)'
        elif 31 <= d_id <= 40:
            return '4_深度分析与预测 (长Prompt)'
        elif 41 <= d_id <= 50:
            return '5_宏观趋势与大局观 (宽泛提问)'
        else:
            return '未知场景'

    df_d['边缘场景'] = df_d['D_ID'].apply(map_d_scenario)

    # 聚合到 Prompt 级别 (一个 D_ID 算作一次提问)
    prompt_level = df_d.groupby(['文件名', '边缘场景']).agg({
        '是否触发搜索': 'first', 
        '引用域名': lambda x: x.notna().sum() # 计算有效引用数量
    }).reset_index()

    # ---------------------------------------------------------
    # 1. 搜索触发率差异 (模糊问题 vs 高风险)
    # ---------------------------------------------------------
    print("\n[核心观察 1] 各极端场景的【主动搜索触发率】:")
    # 假设 CSV 中存在的都算触发了（如果没触发的你已经通过前面的代码找出来了，这里可以用总数 10 来算）
    for scenario in sorted(prompt_level['边缘场景'].unique()):
        # 计算当前场景在数据集中出现了几个不同的 Prompt
        triggered_count = prompt_level[prompt_level['边缘场景'] == scenario]['文件名'].nunique()
        # 每个场景总共有 10 个 Prompt
        trigger_rate = (triggered_count / 10.0) * 100 
        print(f"   - {scenario}: {trigger_rate:.1f}% (触发 {triggered_count}/10)")

    print("   *解读提示: 重点观察‘极度模糊’是否极低，‘高风险’是否被安全策略拦截而不触发。")

    # ---------------------------------------------------------
    # 2. 复杂任务的信息整合能力 (引用丰富度)
    # ---------------------------------------------------------
    print("\n[核心观察 2] 应对复杂任务的【搜索深度/引用广度】:")
    avg_citations = prompt_level.groupby('边缘场景')['引用域名'].mean().round(1)
    for scenario, avg in avg_citations.items():
        print(f"   - {scenario}: 平均抓取 {avg} 个网页进行合成")
    
    print("   *解读提示: ‘多约束复杂任务’理应具有最高的平均引用数，因为模型需要跨多个维度(如价格+距离+评价)查阅资料。")

    # ---------------------------------------------------------
    # 3. 安全策略与来源权威性 (高风险问题)
    # ---------------------------------------------------------
    print("\n[核心观察 3] 安全策略：不同场景引用的【网站权威性(Domain Rank)】:")
    # 取 DR 数据，兼容合并前后的列名
    dr_col = '最终评级(Final_DR)' if '最终评级(Final_DR)' in df_d.columns else '域名评级(Domain Rank)'
    df_d['DR_数值'] = pd.to_numeric(df_d[dr_col], errors='coerce')
    
    dr_stats = df_d.groupby('边缘场景')['DR_数值'].agg(['mean', 'median']).round(1)
    print(dr_stats.to_string())
    print("   *解读提示: ‘高风险’场景的平均 DR 应该显著高于其他场景，说明模型面对医疗金融时，只敢引用最头部的权威机构。")

    # ---------------------------------------------------------
    # 4. 网站类型偏好转移
    # ---------------------------------------------------------
    if '网站类型' in df_d.columns:
        print("\n[核心观察 4] 搜索策略：应对不同场景的【网站类型偏好】:")
        type_crosstab = pd.crosstab(df_d['边缘场景'], df_d['网站类型'], normalize='index') * 100
        
        for scenario in type_crosstab.index:
            top_types = type_crosstab.loc[scenario].sort_values(ascending=False).head(3)
            types_str = ', '.join([f"{t} ({p:.1f}%)" for t, p in top_types.items()])
            print(f"   - {scenario}:\n      首选 -> {types_str}")

    return df_d

# ==========================================
# 运行方法：
# df_d_analyzed = analyze_d_layer_edge_cases(df)
# ==========================================

import os
import csv


def add_prompt_to_csv(input_csv_path: str,
                      output_csv_path: str,
                      prompt_dir: str):
    """
    给已有 CSV 添加 prompt 列

    参数：
        input_csv_path: 原始CSV路径
        output_csv_path: 输出CSV路径
        prompt_dir: prompt文件夹路径
    """

    # -----------------------------
    # 1️⃣ 读取所有 prompt 文件
    # -----------------------------
    prompt_map = {}

    for file in os.listdir(prompt_dir):
        if not file.endswith(".txt"):
            continue

        # promptA_commerce.txt → A_commerce
        name = file.replace("prompt", "").replace(".txt", "")

        path = os.path.join(prompt_dir, file)

        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]

        prompt_map[name] = lines

    # -----------------------------
    # 2️⃣ 读取原 CSV
    # -----------------------------
    rows = []
    with open(input_csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            rows.append(row)

    # -----------------------------
    # 3️⃣ 为每一行添加 prompt
    # -----------------------------
    for row in rows:
        source_file = row.get("source_file", "")

        try:
            parts = source_file.split("_")
            base = "_".join(parts[:-1])   # A_commerce
            idx = int(parts[-1]) - 1      # 第几行（从0开始）

            if base in prompt_map and idx < len(prompt_map[base]):
                row["prompt"] = prompt_map[base][idx]
            else:
                row["prompt"] = ""

        except:
            row["prompt"] = ""

    # -----------------------------
    # 4️⃣ 写新 CSV（加 prompt 列）
    # -----------------------------
    new_fieldnames = fieldnames.copy()
    if "prompt" not in new_fieldnames:
        new_fieldnames.insert(2, "prompt")  # 放在 source_file 后面

    with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Done! Saved to {output_csv_path}")

add_prompt_to_csv(
    input_csv_path="youtube_results.csv",
    output_csv_path="youtube_results_with_prompt.csv",
    prompt_dir="prompt"
)