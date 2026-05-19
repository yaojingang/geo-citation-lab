import os
import csv
import time
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from google import genai

# ================= 配置区域 =================
# 你只需要填主目录，程序会自动穿透找所有子文件夹
HTML_FOLDER_PATH = 'Google/A_commerce/美国'
OUTPUT_CSV = 'analysis_results_A_commerce.csv'

# DataForSEO 认证串
DATAFORSEO_BASE64_AUTH = os.environ.get("DATAFORSEO_BASE64_AUTH", "").strip()

# Gemini API Key (🚨 请填入你的 Key)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# 初始化 Gemini
gemini_client = None
if GEMINI_API_KEY and GEMINI_API_KEY != 'YOUR_NEW_GEMINI_API_KEY':
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ================= 1. HTML 解析 =================

def extract_citations(soup):
    """解析 HTML 提取引用"""
    cited_domains = set()
    citation_pills = soup.find_all(attrs={"data-testid": "webpage-citation-pill"})
    is_search = len(citation_pills) > 0
    
    for pill in citation_pills:
        a_tag = pill.find('a')
        if a_tag and a_tag.get('href'):
            url = a_tag.get('href')
            domain = urlparse(url).netloc.replace('www.', '')
            if domain:
                cited_domains.add(domain)
    return is_search, list(cited_domains)

# ================= 2. DataForSEO 全量数据抓取 =================

def get_dataforseo_technologies(domain):
    """调用 DataForSEO 接口，缺失值全部填 null"""
    data = {
        "DataForSEO_状态": "null",
        "网站标题(Title)": "null",
        "网站描述(Description)": "null",
        "域名评级(Domain Rank)": "null",
        "语言(Language)": "null",
        "国家(Country)": "null",
        "技术栈详情(Technologies)": "null"
    }
    
    if not DATAFORSEO_BASE64_AUTH:
        return data

    headers = {
        'Authorization': f'Basic {DATAFORSEO_BASE64_AUTH}',
        'Content-Type': 'application/json'
    }

    url = "https://api.dataforseo.com/v3/domain_analytics/technologies/domain_technologies/live"
    payload = [{"target": domain}]

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        
        if response.status_code == 200:
            res_json = response.json()
            tasks = res_json.get('tasks', [])
            
            if tasks:
                task = tasks[0]
                if task.get('status_code') == 20000 and task.get('result'):
                    res_data = task['result'][0]
                    data["DataForSEO_状态"] = "成功"
                    
                    data["网站标题(Title)"] = str(res_data.get('title') or 'null')
                    desc = str(res_data.get('description') or 'null')
                    data["网站描述(Description)"] = desc.replace('\n', ' ').replace('\r', '') 
                    data["域名评级(Domain Rank)"] = str(res_data.get('domain_rank') or 'null')
                    data["语言(Language)"] = str(res_data.get('language_code') or 'null')
                    data["国家(Country)"] = str(res_data.get('country_iso_code') or 'null')
                    
                    # 格式化技术栈，缺失的分类填 null
                    tech_data = res_data.get('technologies', {})
                    if isinstance(tech_data, dict) and tech_data:
                        tech_list = [f"{name} [{info.get('category', 'null')}]" for name, info in tech_data.items()]
                        data["技术栈详情(Technologies)"] = " | ".join(tech_list)
                    else:
                        data["技术栈详情(Technologies)"] = "null"
                else:
                    data["DataForSEO_状态"] = "无数据"
            else:
                data["DataForSEO_状态"] = "格式异常"
    except Exception as e:
        print(f"  [DataForSEO 异常] {domain}: {e}")
        data["DataForSEO_状态"] = "网络异常"

    return data

# ================= 3. Gemini 智能分类 =================

def categorize_website(domain):
    if not gemini_client:
        return "null"

    prompt = f"""
你是SEO分析师。
判断域名 {domain} 属于以下哪一类：
新闻 / blog / 行业垂类 / 测评类 / 官网 / 电商 / 其他

只能返回一个词。如果无法判断，请回复 null。
"""
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        result = response.text.strip()
        return result if result else "null"
    except Exception as e:
        print(f"  [Gemini 异常] {domain}: {e}")
        return "null"

# ================= 4. 工具 =================

def extract_sequence_number(filename):
    match = re.search(r'b-[a-f0-9]+-(\d+)', filename, re.IGNORECASE)
    return match.group(1) if match else "null"

# ================= 5. 主流程 =================

def main():
    if not os.path.exists(HTML_FOLDER_PATH):
        print(f"❌ 主文件夹不存在: {HTML_FOLDER_PATH}")
        return

    print("="*50)
    prefix = input("✏️ 请输入你想使用的文件名前缀 (例如 问答_): ")
    print("="*50)

    # 1. 穿透所有子文件夹，找到所有的 html 文件
    html_files = []
    for root, dirs, files in os.walk(HTML_FOLDER_PATH):
        for file in files:
            if file.endswith(('.html', '.htm')):
                # 保存文件的绝对路径
                html_files.append(os.path.join(root, file))
                
    print(f"📂 共在子文件夹中找到 {len(html_files)} 个 HTML 文件，开始执行自动化分析...\n")

    # 定义全空占位符，全部使用 null
    empty_api_data = {
        "DataForSEO_状态": "null", "网站标题(Title)": "null", "网站描述(Description)": "null",
        "域名评级(Domain Rank)": "null", "语言(Language)": "null", "国家(Country)": "null", "技术栈详情(Technologies)": "null"
    }

    keys = [
        "文件名", "是否触发搜索", "引用域名", "网站类型", "DataForSEO_状态", 
        "网站标题(Title)", "网站描述(Description)", "域名评级(Domain Rank)", 
        "语言(Language)", "国家(Country)", "技术栈详情(Technologies)"
    ]

    # 初始化全局域名缓存（跨文件去重，省钱利器）
    domain_cache = {}

    # 提前创建并写入 CSV 表头（实时写入模式）
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

    # 2. 逐个处理文件
    for idx, filepath in enumerate(html_files):
        # 提取文件名（不带路径）用于生成带序号的名字
        filename = os.path.basename(filepath)
        seq = extract_sequence_number(filename)
        name = f"{prefix}{seq}"

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        is_search, domains = extract_citations(soup)
        
        file_results = [] # 暂存当前文件的结果

        if not is_search:
            print(f"⏩ 跳过 [{idx+1}/{len(html_files)}] {name} (未触发搜索)")
            base_dict = {"文件名": name, "是否触发搜索": "否", "引用域名": "null", "网站类型": "null"}
            base_dict.update(empty_api_data)
            file_results.append(base_dict)
            
        elif is_search and not domains:
            base_dict = {"文件名": name, "是否触发搜索": "是", "引用域名": "null", "网站类型": "null"}
            base_dict.update(empty_api_data)
            file_results.append(base_dict)
            
        else:
            for domain in domains:
                # 全局去重：如果缓存里有，直接拿来用
                if domain in domain_cache:
                    print(f"♻️ 命中缓存 [{idx+1}/{len(html_files)}] {name} -> {domain} (跳过API调用)")
                else:
                    print(f"🔍 请求 API [{idx+1}/{len(html_files)}] {name} -> {domain}")
                    tech_data = get_dataforseo_technologies(domain)
                    category = categorize_website(domain)
                    
                    domain_cache[domain] = {
                        "api_data": tech_data,
                        "category": category
                    }
                    time.sleep(1) # 友好停顿防限流

                row = {
                    "文件名": name,
                    "是否触发搜索": "是",
                    "引用域名": domain,
                    "网站类型": domain_cache[domain]["category"]
                }
                row.update(domain_cache[domain]["api_data"])
                file_results.append(row)

        # 当前文件处理完，立刻追加写入 CSV (防崩溃)
        if file_results:
            with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writerows(file_results)

    print(f"\n🎉 完美收工！已成功处理 {len(html_files)} 个文件。全局共分析了 {len(domain_cache)} 个独立域名。")
    print(f"📁 结果已安全保存至：{OUTPUT_CSV}")

if __name__ == "__main__":
    main()
