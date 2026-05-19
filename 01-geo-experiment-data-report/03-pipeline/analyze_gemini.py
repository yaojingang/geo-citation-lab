import os
import csv
import time
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from google import genai

# ================= 配置区域 =================
HTML_FOLDER_PATH = 'Google/D/美国'  # 新文件夹
OUTPUT_CSV = 'analysis_results_D.csv'  # 使用已有 CSV

# DataForSEO 认证串
DATAFORSEO_BASE64_AUTH = os.environ.get("DATAFORSEO_BASE64_AUTH", "").strip()

# Gemini API Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

gemini_client = None
if GEMINI_API_KEY and GEMINI_API_KEY != 'YOUR_NEW_GEMINI_API_KEY':
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ================= 1. HTML 解析 =================
def extract_citations(html_text):
    cited_domains = set()
    is_search = False
    domain_pattern = re.compile(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    soup = BeautifulSoup(html_text, 'html.parser')
    
    a_tags = soup.select('a.NDNGvf')
    if not a_tags:
        aimc = soup.find(attrs={"data-subtree": "aimc"})
        if aimc:
            a_tags = aimc.find_all('a', href=True)
            
    if soup.find(attrs={"data-subtree": "aimc"}) or soup.find(attrs={"data-subtree": "aimfl"}) or a_tags:
        is_search = True
        
    for tag in a_tags:
        url = tag.get('href', '').strip()
        if not url.startswith(('http://', 'https://')):
            continue
        if 'google.com/search' in url or 'gstatic.com' in url or 'googleusercontent.com' in url:
            continue
            
        try:
            domain = urlparse(url).netloc.lower()
            domain = domain.replace('www.', '').split(':')[0]
            if domain and domain_pattern.match(domain):
                cited_domains.add(domain)
        except Exception:
            continue
            
    return is_search, list(cited_domains)

# ================= 2. DataForSEO API =================
def get_dataforseo_technologies(domain):
    data = {
        "DataForSEO_状态": "null", "网站标题(Title)": "null", "网站描述(Description)": "null",
        "域名评级(Domain Rank)": "null", "语言(Language)": "null", "国家(Country)": "null", "技术栈详情(Technologies)": "null"
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

# ================= 3. Gemini 分类 =================
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

def load_existing_cache_from_csv(csv_file):
    cache = {}
    if not os.path.exists(csv_file):
        return cache
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain = row.get("引用域名")
            if domain and domain != 'null':
                cache[domain] = {
                    "api_data": {
                        "DataForSEO_状态": row.get("DataForSEO_状态", "null"),
                        "网站标题(Title)": row.get("网站标题(Title)", "null"),
                        "网站描述(Description)": row.get("网站描述(Description)", "null"),
                        "域名评级(Domain Rank)": row.get("域名评级(Domain Rank)", "null"),
                        "语言(Language)": row.get("语言(Language)", "null"),
                        "国家(Country)": row.get("国家(Country)", "null"),
                        "技术栈详情(Technologies)": row.get("技术栈详情(Technologies)", "null")
                    },
                    "category": row.get("网站类型", "null")
                }
    return cache

# ================= 5. 主流程 =================
def main():
    if not os.path.exists(HTML_FOLDER_PATH):
        print(f"❌ 主文件夹不存在: {HTML_FOLDER_PATH}")
        return

    print("="*50)
    prefix = input("✏️ 请输入你想使用的文件名前缀 (例如 问答_): ")
    print("="*50)

    html_files = []
    for root, dirs, files in os.walk(HTML_FOLDER_PATH):
        for file in files:
            if file.endswith(('.html', '.htm')):
                html_files.append(os.path.join(root, file))
                
    print(f"📂 共在子文件夹中找到 {len(html_files)} 个 HTML 文件，开始执行自动化分析...\n")

    empty_api_data = {
        "DataForSEO_状态": "null", "网站标题(Title)": "null", "网站描述(Description)": "null",
        "域名评级(Domain Rank)": "null", "语言(Language)": "null", "国家(Country)": "null", "技术栈详情(Technologies)": "null"
    }

    keys = [
        "文件名", "是否触发搜索", "引用域名", "网站类型", "DataForSEO_状态", 
        "网站标题(Title)", "网站描述(Description)", "域名评级(Domain Rank)", 
        "语言(Language)", "国家(Country)", "技术栈详情(Technologies)"
    ]

    # 从已有 CSV 加载缓存
    domain_cache = load_existing_cache_from_csv(OUTPUT_CSV)
    if domain_cache:
        print(f"♻️ 已加载 {len(domain_cache)} 个缓存域名，避免重复请求 API\n")

    # 如果 CSV 不存在，先写入表头
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()

    for idx, filepath in enumerate(html_files):
        filename = os.path.basename(filepath)
        seq = extract_sequence_number(filename)
        name = f"{prefix}{seq}"

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            raw_html = f.read()

        is_search, domains = extract_citations(raw_html)
        file_results = [] 

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
                    time.sleep(1) 

                row = {
                    "文件名": name,
                    "是否触发搜索": "是",
                    "引用域名": domain,
                    "网站类型": domain_cache[domain]["category"]
                }
                row.update(domain_cache[domain]["api_data"])
                file_results.append(row)

        if file_results:
            with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writerows(file_results)

    print(f"\n🎉 完美收工！已成功处理 {len(html_files)} 个文件。全局共分析了 {len(domain_cache)} 个独立域名。")
    print(f"📁 结果已安全追加至：{OUTPUT_CSV}")

if __name__ == "__main__":
    main()
