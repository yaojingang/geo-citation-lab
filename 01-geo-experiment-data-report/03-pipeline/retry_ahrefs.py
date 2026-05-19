import os
import csv
import time
import requests
from datetime import datetime, timedelta
from google import genai

# ================= 配置区域 =================
INPUT_CSV = 'analysis_results_debug.csv'   # 读取你上一轮跑完的原始表格
OUTPUT_CSV = 'results_ahrefs_A_technology.csv'          # 单独生成的 Ahrefs 专属表格

# 🚨 请填入你的 API Keys (请务必使用新生成的 Key！)
AHREFS_API_KEY = os.environ.get("AHREFS_API_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

gemini_client = None
if GEMINI_API_KEY and GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY':
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# 为 Ahrefs 量身定制的全新表头
AHREFS_KEYS = [
    "文件名", "引用域名", "网站类型", 
    "DR", "Ahrefs排名(Rank)", "总外链(Backlinks)", 
    "引荐域名(RefDomains)", "自然流量(Org_Traffic)", "自然关键词(Org_Keywords)"
]

# ================= Ahrefs 全量数据接口 =================

def get_ahrefs_full_data(domain):
    """同时调用 Rating 和 Metrics 接口，榨干 Ahrefs 所有指标"""
    data = {
        "DR": "-", "Ahrefs排名(Rank)": "-", 
        "总外链(Backlinks)": "-", "引荐域名(RefDomains)": "-", 
        "自然流量(Org_Traffic)": "-", "自然关键词(Org_Keywords)": "-"
    }

    if not AHREFS_API_KEY or AHREFS_API_KEY == 'YOUR_AHREFS_API_KEY':
        print("  [拦截] 未配置 Ahrefs Key")
        return data

    headers = {
        "Authorization": f"Bearer {AHREFS_API_KEY}",
        "Accept": "application/json"
    }
    
    # 核心修复：强制请求“昨天”的数据，完美避开 Ahrefs 当天数据未生成的时区报错，且满足 date 参数要求
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')

    # 1. 获取 DR 和 全球排名
    try:
        url_dr = f"https://api.ahrefs.com/v3/site-explorer/domain-rating?target={domain}&date={yesterday}"
        res_dr = requests.get(url_dr, headers=headers, timeout=15)
        if res_dr.status_code == 200:
            dr_json = res_dr.json().get('domain_rating', {})
            data["DR"] = str(dr_json.get('domain_rating') or '-')
            data["Ahrefs排名(Rank)"] = str(dr_json.get('ahrefs_rank') or '-')
        else:
            print(f"  ⚠️ [DR接口异常] 状态码: {res_dr.status_code} | 信息: {res_dr.text}")
    except Exception as e:
        print(f"  ❌ [DR网络异常] {domain}: {e}")

    # 2. 获取流量、外链等核心 Metrics
    try:
        url_metrics = f"https://api.ahrefs.com/v3/site-explorer/metrics?target={domain}&date={yesterday}"
        res_metrics = requests.get(url_metrics, headers=headers, timeout=15)
        if res_metrics.status_code == 200:
            m_json = res_metrics.json().get('metrics', {})
            data["总外链(Backlinks)"] = str(m_json.get('backlinks') or '-')
            data["引荐域名(RefDomains)"] = str(m_json.get('refdomains') or '-')
            data["自然流量(Org_Traffic)"] = str(m_json.get('org_traffic') or '-')
            data["自然关键词(Org_Keywords)"] = str(m_json.get('org_keywords') or '-')
        else:
            print(f"  ⚠️ [Metrics接口异常] 状态码: {res_metrics.status_code} | 信息: {res_metrics.text}")
    except Exception as e:
        print(f"  ❌ [Metrics网络异常] {domain}: {e}")

    return data

# ================= Gemini 分类 =================

def categorize_website(domain):
    """仅在万不得已时调用，增加超时保护"""
    if not gemini_client:
        return "null"
    prompt = f"你是SEO分析师。判断域名 {domain} 属于以下哪一类：新闻 / blog / 行业垂类 / 测评类 / 官网 / 电商 / 其他。只能返回一个词。如果无法判断，请回复 null。"
    try:
        response = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        result = response.text.strip()
        return result if result else "null"
    except Exception as e:
        print(f"  [Gemini 限流/异常] {domain}: {e}")
        return "null"

# ================= 主流程 =================

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 找不到原始文件: {INPUT_CSV}")
        return

    print("="*50)
    print("🚀 开始读取原始 CSV，筛选需要 Ahrefs 降维打击的大站...")
    print("="*50)

    rescue_rows = []
    
    # 使用字典来存储去重后的域名，同时保存它们在原表中已有的分类！
    # 结构: {"domain.com": "新闻", "domain2.com": "官网"}
    rescue_domains = {}

    with open(INPUT_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row.get("DataForSEO_状态", "")
            domain = row.get("引用域名", "")
            is_search = row.get("是否触发搜索", "")
            category = row.get("网站类型", "null")
            
            # 只要是没有拿到数据的，全部纳入抢救范围
            if status not in ["成功", "获取成功"] and is_search == "是" and domain and domain != "null":
                rescue_rows.append(row)
                # 如果这个域名还没被加入待抢救名单，或者它之前的分类是 null，就更新它的分类
                if domain not in rescue_domains or rescue_domains[domain] == "null":
                    rescue_domains[domain] = category

    if not rescue_domains:
        print("✅ 扫描完毕：没有发现需要抢救的漏网之鱼！")
        return

    domain_list = list(rescue_domains.keys())
    print(f"⚠️ 锁定 {len(domain_list)} 个目标域名，启动 Ahrefs 数据引擎...\n")

    rescue_cache = {}
    
    for idx, domain in enumerate(domain_list):
        print(f"🔄 [{idx+1}/{len(domain_list)}] 查询 Ahrefs: {domain} ... ", end="", flush=True)
        
        ahrefs_data = get_ahrefs_full_data(domain)
        
        # 核心修复：从输入文件继承分类，不再无脑请求 Gemini！
        existing_category = rescue_domains[domain]
        if existing_category and existing_category != "null":
            category = existing_category
        else:
            # 只有原表真的没分出来，才请求一次大模型
            category = categorize_website(domain)
        
        if ahrefs_data["DR"] != "-":
            print(f"✅ 成功! DR: {ahrefs_data['DR']} | 流量: {ahrefs_data['自然流量(Org_Traffic)']}")
        else:
            print("❌ 获取失败")
            
        rescue_cache[domain] = {
            "api_data": ahrefs_data,
            "category": category
        }
        # 停顿1.5秒，确保不触发 Ahrefs 的并发限制
        time.sleep(1.5) 

    final_ahrefs_results = []
    for row in rescue_rows:
        domain = row.get("引用域名")
        
        new_row = {
            "文件名": row.get("文件名", ""),
            "引用域名": domain,
            "网站类型": rescue_cache[domain]["category"]
        }
        new_row.update(rescue_cache[domain]["api_data"])
        final_ahrefs_results.append(new_row)

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=AHREFS_KEYS)
        writer.writeheader()
        writer.writerows(final_ahrefs_results)

    print("="*50)
    print(f"🎉 降维打击完成！成功处理了 {len(domain_list)} 个巨头网站。")
    print(f"📁 结果已安全输出至：{OUTPUT_CSV}")
    print("="*50)

if __name__ == "__main__":
    main()
