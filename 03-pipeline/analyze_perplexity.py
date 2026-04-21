import os
import csv
import time
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from google import genai

# ================= 配置区域 =================
HTML_FOLDER_PATH = 'perplexity/D/美国'  # 替换为你的 Perplexity 根文件夹
OUTPUT_CSV = 'perplexity_analysis_results.csv'     # 输出文件
CHATGPT_CSV = 'chatgpt_results.csv'                # 需要交叉比对的 ChatGPT 结果文件

# DataForSEO 认证串
DATAFORSEO_BASE64_AUTH = os.environ.get("DATAFORSEO_BASE64_AUTH", "").strip()

# Gemini API Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ================= 1. Perplexity HTML 解析 =================
def extract_citations_perplexity(html_text):
    cited_domains = set()
    domain_pattern = re.compile(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    soup = BeautifulSoup(html_text, 'html.parser')
    
    a_tags = soup.find_all('a', href=True)
    is_search = len(a_tags) > 0
        
    for tag in a_tags:
        url = tag.get('href', '').strip()
        if not url.startswith(('http://', 'https://')):
            continue
            
        # 排除内部系统链接
        internal_patterns = ['perplexity.ai', 'google.com', 'gstatic.com', 'googleusercontent.com', 'facebook.com/tr', 'doubleclick.net']
        if any(pat in url.lower() for pat in internal_patterns):
            continue
            
        try:
            domain = urlparse(url).netloc.lower()
            domain = domain.replace('www.', '').split(':')[0]
            if domain and domain_pattern.match(domain):
                cited_domains.add(domain)
        except Exception:
            continue
            
    return is_search, list(cited_domains)

# ================= 2. API 调用逻辑 =================
def get_dataforseo_technologies(domain):
    data = {
        "DataForSEO_状态": "null", "网站标题(Title)": "null", "网站描述(Description)": "null",
        "域名评级(Domain Rank)": "null", "语言(Language)": "null", "国家(Country)": "null", "技术栈详情(Technologies)": "null",
        "Ahrefs排名(Rank)": "null", "总外链(Backlinks)": "null", "引荐域名(RefDomains)": "null", 
        "自然流量(Org_Traffic)": "null", "自然关键词(Org_Keywords)": "null", "最终评级(Final_DR)": "null", "评级数据来源": "DataForSEO"
    }
    if not DATAFORSEO_BASE64_AUTH: return data
    headers = {'Authorization': f'Basic {DATAFORSEO_BASE64_AUTH}', 'Content-Type': 'application/json'}
    url = "https://api.dataforseo.com/v3/domain_analytics/technologies/domain_technologies/live"
    payload = [{"target": domain}]
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            res_json = response.json()
            tasks = res_json.get('tasks', [])
            if tasks and tasks[0].get('status_code') == 20000 and tasks[0].get('result'):
                res_data = tasks[0]['result'][0]
                domain_rank = str(res_data.get('domain_rank') or 'null')
                data.update({
                    "DataForSEO_状态": "成功",
                    "网站标题(Title)": str(res_data.get('title') or 'null'),
                    "网站描述(Description)": str(res_data.get('description') or 'null').replace('\n', ' '),
                    "域名评级(Domain Rank)": domain_rank,
                    "最终评级(Final_DR)": domain_rank, # 默认取 DataForSeo 的 Rank 作为 Final_DR
                    "语言(Language)": str(res_data.get('language_code') or 'null'),
                    "国家(Country)": str(res_data.get('country_iso_code') or 'null')
                })
                tech_data = res_data.get('technologies', {})
                if isinstance(tech_data, dict) and tech_data:
                    data["技术栈详情(Technologies)"] = " | ".join([f"{n} [{i.get('category', 'null')}]" for n, i in tech_data.items()])
    except Exception as e:
        print(f"  [DataForSEO 异常] {domain}: {e}")
    return data

def categorize_website(domain):
    if not gemini_client: return "null"
    prompt = f"你是SEO分析师。判断域名 {domain} 属于以下哪一类：新闻 / blog / 行业垂类 / 测评类 / 官网 / 电商 / 其他。只能返回一个词。如果无法判断，请回复 null。"
    try:
        response = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text.strip() or "null"
    except: return "null"

# ================= 3. 工具函数与缓存加载 =================
def extract_perplexity_info(filename):
    match = re.search(r'b-[a-f0-9]+-(\d+)_(\d+)_', filename)
    return (match.group(1), match.group(2)) if match else ("null", "null")

# 通用 CSV 读取，用于构建缓存字典
def load_csv_to_cache(csv_file):
    cache = {}
    if os.path.exists(csv_file):
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                domain = row.get("引用域名")
                if domain and domain != 'null':
                    # 【修复关键点】：增加 k is not None 过滤掉脏列
                    cached_data = {
                        k: v for k, v in row.items() 
                        if k is not None and k not in ["文件名", "是否触发搜索", "引用域名"]
                    }
                    cache[domain] = cached_data
    return cache

# ================= 4. 主流程 =================
def main():
    print("="*60)
    prefix = input("✏️ 请输入你想使用的文件名前缀: ")
    print("="*60)
    
    # 统一输出表头（融合了两个表格的所有字段）
    keys = [
        "文件名", "是否触发搜索", "引用域名", "DataForSEO_状态", "网站标题(Title)", 
        "网站描述(Description)", "域名评级(Domain Rank)", "语言(Language)", 
        "国家(Country)", "技术栈详情(Technologies)", "Ahrefs排名(Rank)", 
        "总外链(Backlinks)", "引荐域名(RefDomains)", "自然流量(Org_Traffic)", 
        "自然关键词(Org_Keywords)", "最终评级(Final_DR)", "评级数据来源", "网站类型"
    ]
    
    # 加载已有的 Perplexity 结果（断点续传）
    local_cache = load_csv_to_cache(OUTPUT_CSV)
    if local_cache:
        print(f"♻️ 从 {OUTPUT_CSV} 中加载了 {len(local_cache)} 个本地域名缓存")

    # 加载 ChatGPT 结果库（全局白嫖库）
    chatgpt_cache = load_csv_to_cache(CHATGPT_CSV)
    if chatgpt_cache:
        print(f"🤝 从 {CHATGPT_CSV} 中加载了 {len(chatgpt_cache)} 个 ChatGPT 共享域名数据")

    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            csv.DictWriter(f, fieldnames=keys).writeheader()

    target_files = []
    for root, _, files in os.walk(HTML_FOLDER_PATH):
        for file in files:
            if not file.endswith(('.html', '.htm')): continue
            seq, file_idx = extract_perplexity_info(file)
            if file_idx == "1":  # 只提取引用页
                target_files.append((os.path.join(root, file), seq))

    print(f"\n📂 共筛选到 {len(target_files)} 个引用页文件开始处理...\n")

    for filepath, seq in target_files:
        name = f"{prefix}{seq}"
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            raw_html = f.read()

        is_search, domains = extract_citations_perplexity(raw_html)

        rows = []
        if not domains:
            print(f"⏩ 跳过 [{name}] (未提取到有效域名)")
            row = {"文件名": name, "是否触发搜索": "是" if is_search else "否", "引用域名": "null"}
            row.update({k: "null" for k in keys if k not in row})
            rows.append(row)
        else:
            print(f"📄 处理 [{name}]: 提取到 {len(domains)} 个域名")
            for domain in domains:
                row = {"文件名": name, "是否触发搜索": "是", "引用域名": domain}
                
                # 优先级 1: 本地缓存
                if domain in local_cache:
                    print(f"  [缓存命中] -> {domain}")
                    row.update(local_cache[domain])
                
                # 优先级 2: ChatGPT 数据库 
                elif domain in chatgpt_cache:
                    print(f"  [ChatGPT库命中] -> {domain} (白嫖成功)")
                    # 从 ChatGPT 的字段结构，对其拷贝
                    chatgpt_data = chatgpt_cache[domain]
                    for k in keys[3:]: # 将数据填入
                        row[k] = chatgpt_data.get(k, "null")
                    
                    # 放入本地缓存，防止下次重复判断
                    local_cache[domain] = {k: row[k] for k in keys[3:]}

                # 优先级 3: 走 API
                else:
                    print(f"  [请求 API] -> {domain}")
                    api_data = get_dataforseo_technologies(domain)
                    category = categorize_website(domain)
                    
                    # 组装新请求来的数据
                    for k in keys[3:]:
                        if k == "网站类型":
                            row[k] = category
                        elif k in api_data:
                            row[k] = api_data[k]
                        else:
                            row[k] = "null"
                            
                    # 更新本地缓存
                    local_cache[domain] = {k: row[k] for k in keys[3:]}
                    time.sleep(0.5)

                rows.append(row)

        if rows:
            with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writerows(rows)

    print(f"\n🎉 完美收工！结果已存入 {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
