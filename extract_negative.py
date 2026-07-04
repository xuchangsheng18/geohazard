import shutil
from pathlib import Path


def main():
    # 路径配置（确保在项目根目录下运行）
    source_dir = Path("data/negative_samples")
    target_dir = Path("data/raw")

    if not source_dir.exists():
        print("❌ 找不到 negative_samples 文件夹，请确认路径！")
        return

    print("🚀 开始一键物理合并数据集...")

    # 遍历 train 和 valid 文件夹
    for split in ["train", "valid"]:
        # 遍历要合并的子文件夹类型
        for data_type in ["images", "labels", "reports"]:
            src = source_dir / split / data_type
            dst = target_dir / split / data_type

            if src.exists():
                dst.mkdir(parents=True, exist_ok=True)
                files = list(src.iterdir())
                if len(files) > 0:
                    print(f"📦 正在移动 {split}/{data_type} 里的 {len(files)} 个文件...")
                    for f in files:
                        # 使用 move 直接剪切过去，瞬间完成且不占双倍硬盘空间
                        shutil.move(str(f), str(dst / f.name))

    print("\n🧹 正在清理空的 negative_samples 文件夹...")
    shutil.rmtree(source_dir, ignore_errors=True)

    print("\n🎉 大功告成！正负样本已完美融合！")
    print("👉 下一步：直接敲击 python train_model.py --config configs/default.yaml 起飞！")


if __name__ == "__main__":
    main()