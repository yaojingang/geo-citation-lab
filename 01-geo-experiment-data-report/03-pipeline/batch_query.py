import os
import requests
import time
import json

# 1. 基础配置 
BASE_URL = os.environ.get("BATCH_API_BASE_URL", "http://188.166.211.11:9000").rstrip("/")
TOKEN = os.environ.get("BATCH_API_TOKEN", "").strip()


def _headers():
    if not TOKEN:
        raise RuntimeError("Missing BATCH_API_TOKEN. Export it before running batch_query.py.")
    return {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }

def submit_batch():
    """第一步：读取本地 TXT 文件并提交批量任务"""
    print("正在读取 promptD.txt ...")
    
    # 1. 从 TXT 文件中读取所有的 prompt
    queries = []
    with open("prompt/promptD.txt", "r", encoding="utf-8") as file:
        for line in file:
            clean_line = line.strip() # 去除前后的空格和换行符
            if clean_line:            # 如果这一行不是空的，就加到列表里
                queries.append(clean_line)
                
    print(f"成功读取了 {len(queries)} 个 prompt！正在提交...")

    # 2. 构造请求体发送给服务器
    url = f"{BASE_URL}/api/batch"
    payload = {
        "queries": queries, # <--- 这里直接使用读取到的列表
        "countries": ["美国"],
        "platforms": ["perplexity"],
    }
    
    response = requests.post(url, headers=_headers(), json=payload)
    response.raise_for_status()
    data = response.json()
    
    print(f"✅ 提交成功！批次ID: {data['batchId']}, 预计将产生任务数: {data['total']}")
    return data['batchId']

def check_status(batch_id):
    """第二步：查询任务进度"""
    url = f"{BASE_URL}/api/batch/{batch_id}"
    response = requests.get(url, headers=_headers())
    response.raise_for_status()
    return response.json()

def download_results(batch_id, save_path="results.zip", force=False):
    """第三步：下载结果"""
    print(f"\n正在下载结果到 {save_path} ...")
    
    # 如果开启 force=True，可以在未完成时增量下载
    url = f"{BASE_URL}/api/batch/{batch_id}/download"
    if force:
        url += "?force=true"
        
    response = requests.get(url, headers=_headers(), stream=True)
    
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ 下载完成！")
    else:
        print(f"❌ 下载失败，状态码: {response.status_code}, 返回信息: {response.text}")

# ==========================================
# 主程序逻辑
# ==========================================
if __name__ == "__main__":
    try:
        # 1. 提交任务获取 ID
        batch_id = submit_batch()
        
        # 2. 轮询等待任务完成 (每 5 秒查一次)
        while True:
            time.sleep(5) 
            status_data = check_status(batch_id)
            status = status_data.get("status")
            completed = status_data.get("completed", 0)
            total = status_data.get("total", 0)
            pending = status_data.get("pending", 0)
            failed = status_data.get("failed", 0)
            
            print(f"进度更新 -> 状态: {status} | 完成: {completed}/{total} | 排队: {pending} | 失败: {failed}")
            
            if status == "completed":
                print("🎉 所有任务已达到终态！")
                break
                
        # 3. 任务完成后下载压缩包
        download_results(batch_id, save_path=f"results_{batch_id}.zip")
        
    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {e}")
