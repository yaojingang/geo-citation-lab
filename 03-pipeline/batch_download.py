import requests
import os

# ================= 基础配置 =================
BASE_URL = os.environ.get("BATCH_API_BASE_URL", "http://188.166.211.11:9000").rstrip("/")
TOKEN = os.environ.get("BATCH_API_TOKEN", "").strip()


def _headers():
    if not TOKEN:
        raise RuntimeError("Missing BATCH_API_TOKEN. Export it before running batch_download.py.")
    return {"Authorization": f"Bearer {TOKEN}"}

# ================= 填入您的 Batch IDs =================
# 在这个列表里用引号把您的 batchId 括起来，用逗号隔开
BATCH_IDS = [
    "b-e3b2b8e6", #perplexity A_finance
    "b-0160b4fe", #perplexity A_healthcare
    "b-4f043aec", #perplexity A_local
    "b-d2d17691", #perplexity A_news
    "b-e7f21ec1", #perplexity B
    "b-95c3ea39", #perplexity C
    "b-a843542c" #perplexity D
]

def download_batch(batch_id):
    """下载单个批次的结果"""
    print(f"\n正在尝试下载批次: {batch_id} ...")
    
    # 默认使用 force=true，确保能把部分完成的结果也拉下来
    url = f"{BASE_URL}/api/batch/{batch_id}/download?force=true"
    
    try:
        response = requests.get(url, headers=_headers(), stream=True)
        
        if response.status_code == 200:
            save_path = f"results_{batch_id}.zip"
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"✅ 下载成功！文件已保存为: {save_path}")
            
        elif response.status_code == 400:
            print(f"⚠️ 下载跳过 (400): {response.text}")
            print("   -> 原因：这个批次可能还没有任何任务完成落盘，暂无文件可下。")
            
        elif response.status_code == 404:
            print(f"❌ 下载失败 (404): 找不到该批次，请检查 ID 是否拼写正确。")
            
        else:
            print(f"❌ 下载失败，状态码 {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ 网络请求异常: {e}")

if __name__ == "__main__":
    # 过滤掉可能不小心填入的空字符串
    valid_ids = [bid.strip() for bid in BATCH_IDS if bid.strip()]
    
    print(f"共检测到 {len(valid_ids)} 个待下载的批次 ID。")
    print("=" * 50)
    
    for batch_id in valid_ids:
        download_batch(batch_id)
        
    print("\n🎉 所有下载任务执行完毕！")
