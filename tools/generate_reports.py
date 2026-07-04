import os
import time
import base64
from pathlib import Path
from openai import OpenAI

# ==========================================
# ⚙️ 配置区域
# ==========================================
# 请替换为你的阿里云 DashScope API Key
# 我这里用的是硅基流动
API_KEY = "你的api密钥"
BASE_URL = "https://api.siliconflow.cn/v1"

# 使用的模型：qwen-vl-max (效果最好) 或 qwen-vl-plus (性价比高)
MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"

# 数据集根目录
DATA_ROOT = Path("../data/raw")
SPLITS = ["train", "valid", "test"]

# 允许的图片扩展名
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ==========================================
# 🛠️ 核心函数
# ==========================================

def encode_image(image_path: Path) -> str:
    """将本地图片读取为 Base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_mime_type(image_path: Path) -> str:
    """获取图片的 MIME 类型"""
    ext = image_path.suffix.lower()
    if ext in [".jpg", ".jpeg"]: return "image/jpeg"
    if ext == ".png": return "image/png"
    if ext == ".webp": return "image/webp"
    return "image/jpeg"


def generate_report(client: OpenAI, image_path: Path) -> str:
    """调用 Qwen-VL API 生成地质灾害勘查报告"""
    base64_image = encode_image(image_path)
    mime_type = get_image_mime_type(image_path)

    # 构建 Prompt：指导模型生成对多模态训练最有帮助的语义
    system_prompt = """你是一名专业的地质灾害现场勘查专家。请根据提供的现场图片，撰写一段简洁但语义丰富的现场勘查报告（100-200字即可）。

    重点关注并描述以下内容：
    1. 灾害类型与特征：画面中是否存在倒伏树木(fallen tree)、滑坡(landslide)、道路塌陷(road collapse)或落石(stone)？灾害的严重程度如何？
    2. 环境背景：天气、光线、地形地貌（如山体、公路、植被等）。
    3. 人员与设备：画面中是否有施工人员、巡检人员、安全帽、车辆或工程机械？（这非常重要，有助于排除误报）。

    请直接输出报告正文，语气专业、客观，不需要任何开场白或多余的解释。"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.2,  # 低温度保证报告的客观性和稳定性
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\n❌ API 请求失败 {image_path.name}: {e}")
        return ""


# ==========================================
# 🚀 执行主流程
# ==========================================

def main():
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    total_images = 0
    processed_images = 0

    print(f"🔍 开始扫描数据集目录: {DATA_ROOT.absolute()}")

    for split in SPLITS:
        img_dir = DATA_ROOT / split / "images"
        rpt_dir = DATA_ROOT / split / "reports"

        if not img_dir.exists():
            print(f"⚠️ 找不到目录，已跳过: {img_dir}")
            continue

        # 确保 reports 文件夹存在
        rpt_dir.mkdir(parents=True, exist_ok=True)

        # 收集图片文件
        images = [f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTS]
        total_images += len(images)

        print(f"\n📂 处理数据划分: {split} (共 {len(images)} 张图片)")

        for idx, img_path in enumerate(images, 1):
            report_path = rpt_dir / f"{img_path.stem}.txt"

            # 断点续传逻辑：如果报告已存在，则跳过
            if report_path.exists():
                print(f"  [{idx}/{len(images)}] ⏭️ 跳过已存在: {report_path.name}")
                continue

            print(f"  [{idx}/{len(images)}] 📝 正在生成报告: {img_path.name} ...", end="", flush=True)

            report_content = generate_report(client, img_path)

            if report_content:
                # 写入文本文件
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(report_content)
                print(" ✅ 完成")
                processed_images += 1

            # 基础限流保护：避免触发 API 的 QPS 限制（根据你的账号等级可调整）
            time.sleep(0.5)

    print("\n🎉 全部处理完毕！")
    print(f"总计图片数: {total_images}")
    print(f"本次新生成报告数: {processed_images}")


if __name__ == "__main__":
    main()
